"""
Role: Attach monthly environmental covariates to the scored survey
table.
Description: Joins the scored survey table to the monthly covariate
table on the child identifiers, cluster_number, household_number,
hvidx, and survey_round, so each child receives its own twelve
monthly rows. The coordinates in the two tables come from the same
points and are checked to agree, not used to match. Rows with
coordinates of 0, 0 are dropped, mri_value is dropped from the
covariates when present, and each survey round is renamed to its
first year. Writes one CSV in long form, one row per child and
month.
Author: Bonny
"""

import logging
import os
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pathlib import Path
from scipy.special import expit, logit
from scipy.stats import norm
from typing import NamedTuple

log = logging.getLogger(__name__)

COORDINATE_TOLERANCE_DEG = 1e-6

# Cluster and household numbers restart each survey round, so a child is
# identified by all four columns together.
CHILD_KEY = ['cluster_number', 'household_number', 'hvidx', 'survey_round']

# What prepare_data does with a child that appears twice in the survey.
SURVEY_REPEAT_POLICIES = ('raise', 'collapse')

SURVEY_COLUMNS = [
    'cluster_number', 'household_number', 'hvidx', 'mother_caseid',
    'survey_round', 'LATNUM', 'LONGNUM', 'malaria_risk_score',
    'malaria_risk_category', 'malaria_rdt_result',
]

COVARIATE_COLUMNS = [
    'year', 'longitude', 'latitude', 'ndvi', 'month', 
    'precipitation_mm', 'temperature_C',
]

FINAL_COLUMNS = COVARIATE_COLUMNS + ['month_num',
    'cluster_number', 'household_number', 'hvidx', 'mother_caseid',
    'survey_round', 'malaria_risk_score', 'malaria_risk_category',
    'malaria_rdt_result', 
]

MONTH_NUMBER = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11,
    'dec': 12,
}

STATE_COVARIATES = [
    'ndvi', 'precipitation_mm', 'temperature_C', 'elevation_m',
]

RISK_CATEGORIES = [
    'extremely low risk', 'low risk', 'high risk', 'extremely high risk',
]
 
SCORE_CLIP = 0.001

def _require_columns(
    frame: pd.DataFrame, required: list[str], name: str
) -> None:
    """
    This function checks that a table has every column:

    Parameters
    ----------
    frame : pd.DataFrame.
        The table to check.
    required : list[str].
        Column names that must be present.
    name : str.
        Name of the table, used in the error message.

    Returns
    -------
    None
            Nothing is returned. Raises ValueError naming the missing
            columns, so a wrong file fails here and not deep in a
            join.

    """
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f'{name} has no {missing} column(s)')


def _collapse_repeated_survey_children(
    survey: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep one row for each child that appears more than once:

    Parameters
    ----------
    survey : pd.DataFrame.
        The scored survey table, with the CHILD_KEY columns and
        mother_caseid.

    Returns
    -------
    result : pd.DataFrame
            The table with the first row kept for every child and
            the later copies skipped. Copies of a child must agree
            on every column except mother_caseid, as the risk score,
            category, coordinates, and RDT result describe the same
            child. When the copies disagree on mother_caseid, the
            row kept has it set to missing, since choosing one would
            be a guess. The rows skipped, and the children affected,
            are logged as a warning. Raises ValueError when copies
            disagree on any other column, naming the columns, since
            skipping one would then discard real information.

    """
    repeated = survey.duplicated(subset=CHILD_KEY, keep=False)
    if not repeated.any():
        return survey

    others = [
        c for c in survey.columns
        if c not in CHILD_KEY and c != 'mother_caseid'
    ]
    copies = survey[repeated].groupby(CHILD_KEY)
    spread = copies[others].nunique(dropna=False)
    conflicting = (spread > 1).any(axis=1)
    if conflicting.any():
        columns = spread.columns[(spread[conflicting] > 1).any()].tolist()
        raise ValueError(
            f'{int(conflicting.sum())} repeated child(ren) have copies '
            f'that disagree on {columns}'
        )

    mothers = copies['mother_caseid'].nunique(dropna=False)
    ambiguous = mothers[mothers > 1].index
    kept = survey.drop_duplicates(subset=CHILD_KEY, keep='first').copy()
    unclear = kept.set_index(CHILD_KEY).index.isin(ambiguous)
    kept.loc[unclear, 'mother_caseid'] = np.nan

    affected = survey.loc[repeated, CHILD_KEY].drop_duplicates()
    log.warning(
        'skipped %s repeated row(s) of %s child(ren) in the survey '
        'table; %s had copies with different mother_caseid, now set '
        'to missing. First few: %s',
        len(survey) - len(kept), len(affected), int(unclear.sum()),
        affected.head(6).to_dict('records'),
    )
    return kept


def prepare_data(
    survey: pd.DataFrame,
    covariates: pd.DataFrame,
    repeats: str = 'raise',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean the survey and covariate tables before matching:

    Parameters
    ----------
    survey : pd.DataFrame.
        The scored survey table, one row per child.
    covariates : pd.DataFrame.
        The monthly covariate table, one row per child and month.
    repeats : str.
        What to do when a child appears more than once in the
        survey table. 'raise' stops with an error. 'collapse' keeps
        the first copy and skips the others, with a warning, see
        _collapse_repeated_survey_children.

    Returns
    -------
    result : tuple[pd.DataFrame, pd.DataFrame]
            The survey table and the covariate table. Rows of the
            survey table with LATNUM and LONGNUM both 0 are dropped
            and counted in the log. survey_round is read as text in
            both tables, so a table holding one round whose label
            looks like a number still joins to the other. The
            mri_value column is dropped from the covariates when
            present. Raises ValueError when a child appears more
            than once in the survey table, or a child and month more
            than once in the covariate table, since the join would
            then multiply rows.

    """
    if repeats not in SURVEY_REPEAT_POLICIES:
        raise ValueError(
            f'repeats must be one of {SURVEY_REPEAT_POLICIES}, got '
            f'{repeats!r}'
        )

    survey = survey.copy()
    covariates = covariates.copy()
    for table in (survey, covariates):
        table['survey_round'] = table['survey_round'].astype(str)

    at_origin = (survey['LATNUM'] == 0) & (survey['LONGNUM'] == 0)
    if at_origin.any():
        log.info(
            'dropping %s row(s) with coordinates 0, 0',
            int(at_origin.sum()),
        )
    survey = survey.loc[~at_origin].copy()

    if 'mri_value' in covariates.columns:
        log.info('dropping mri_value from the covariates')
        covariates = covariates.drop(columns=['mri_value'])

    if repeats == 'collapse':
        survey = _collapse_repeated_survey_children(survey)

    repeated = survey.duplicated(subset=CHILD_KEY)
    if repeated.any():
        raise ValueError(
            f'{int(repeated.sum())} row(s) of the survey table repeat '
            f'a child already seen, identified by {CHILD_KEY} '
            "(repeats='collapse' keeps one copy of each repeated child)"
        )
    repeated = covariates.duplicated(subset=CHILD_KEY + ['month'])
    if repeated.any():
        raise ValueError(
            f'{int(repeated.sum())} row(s) of the covariate table '
            f'repeat a child and month already seen, identified by '
            f'{CHILD_KEY} and month'
        )
    return survey, covariates


def merge_covariates(
    survey: pd.DataFrame, covariates: pd.DataFrame
) -> pd.DataFrame:
    """
    Join each child to its own monthly covariates:

    Parameters
    ----------
    survey : pd.DataFrame.
        The survey table returned by prepare_data.
    covariates : pd.DataFrame.
        The covariate table returned by prepare_data.

    Returns
    -------
    result : pd.DataFrame
            One row per child and month. The match is exact, on
            CHILD_KEY, and no distance is involved. Raises
            ValueError when a child has no row in the covariates, so
            two files that are out of step fail here and not as a
            shorter table, or when the survey coordinates and the
            covariate coordinates of a matched child differ by more
            than COORDINATE_TOLERANCE_DEG, since a child matched to
            the wrong location would otherwise go unnoticed. A
            missing covariate value is kept as missing and logged
            as a warning.

    """
    merged = survey.merge(
        covariates[CHILD_KEY + COVARIATE_COLUMNS],
        on=CHILD_KEY,
        how='left',
        indicator=True,
        validate='one_to_many',
    )
    unmatched = int((merged['_merge'] == 'left_only').sum())
    if unmatched:
        raise ValueError(
            f'{unmatched} child(ren) of the survey table have no row '
            'in the covariates'
        )
    merged = merged.drop(columns=['_merge'])

    differs = (
        (merged['LONGNUM'] - merged['longitude']).abs()
        > COORDINATE_TOLERANCE_DEG
    ) | (
        (merged['LATNUM'] - merged['latitude']).abs()
        > COORDINATE_TOLERANCE_DEG
    )
    if differs.any():
        children = len(merged.loc[differs, CHILD_KEY].drop_duplicates())
        raise ValueError(
            f'{children} child(ren) have survey coordinates that '
            f'differ from their covariate coordinates by more than '
            f'{COORDINATE_TOLERANCE_DEG} degrees'
        )

    for column in ('ndvi', 'precipitation_mm', 'temperature_C'):
        missing = int(merged[column].isna().sum())
        if missing:
            log.warning('%s row(s) have no %s value', missing, column)

    merged['month_num'] = _month_number(merged['month'])
    log.info(
        '%s child(ren) matched exactly to %s monthly row(s)',
        len(survey), len(merged),
    )
    return merged


def build_survey_covariate_table(
    survey_path: str,
    covariate_path: str,
    output_path: str,
    overwrite: bool = False,
    repeats: str = 'raise',
) -> str:
    """
    Write the survey table joined to its monthly covariates:

    Parameters
    ----------
    survey_path : str.
        Path of the scored survey CSV, for example the output of
        merge_risk_with_outcome.
    covariate_path : str.
        Path of the monthly covariate CSV, one row per child and
        month, with the same child identifiers as the survey CSV.
    output_path : str.
        Path of the CSV file to write.
    overwrite : bool.
        When False and output_path already exists, the existing file
        is left as is and the inputs are not read again. When True,
        it is regenerated.
    repeats : str.
        Passed to prepare_data: what to do when a child appears more
        than once in the survey table, 'raise' or 'collapse'.

    Returns
    -------
    result : str
            output_path. The columns are FINAL_COLUMNS, one row per
            child and month. survey_round holds the first year of
            each round as an integer, for example 2014 for 2014-15,
            the same year as the covariate rows in year.

    """
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        log.info('%s already present, nothing to do', output_path)
        return output_path

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    survey = pd.read_csv(survey_path, low_memory=False)
    covariates = pd.read_csv(covariate_path, low_memory=False)
    _require_columns(survey, SURVEY_COLUMNS, survey_path)
    _require_columns(
        covariates, CHILD_KEY + COVARIATE_COLUMNS, covariate_path
    )

    survey, covariates = prepare_data(survey, covariates, repeats)
    merged = merge_covariates(survey, covariates)

    # The join needs the labels as they are in both files. The rename
    # to the first year comes after it.
    merged['survey_round'] = merged['survey_round'].map(
        _first_survey_year
    )
    result = merged[FINAL_COLUMNS]

    result.to_csv(output_path, index=False)
    log.info(
        'wrote %s with %s row(s) and %s column(s)',
        output_path, len(result), result.shape[1],
    )
    return output_path


class YearData(NamedTuple):
    """
    The arrays the state space model needs for one survey year:
 
    Attributes
    ----------
    child_index : np.ndarray.
        For each row of the year's table, the position of its child.
    month_index : np.ndarray.
        For each row, the month from 0 (January) to 11 (December).
    child_point : np.ndarray.
        For each child, the position of its matched grid point.
    annual_logit : np.ndarray.
        For each child, the logit of the annual malaria_risk_score.
    point_count : np.ndarray.
        For each point, the number of children.
    point_mean : np.ndarray.
        For each point, the mean annual logit of its children.
    point_spread : np.ndarray.
        For each point, the sum of squared differences of its
        children's annual logits from that mean.
    covariates : np.ndarray.
        Standardized covariates with shape points, months, covariates,
        in the order of STATE_COVARIATES.
 
    """
 
    child_index: np.ndarray
    month_index: np.ndarray
    child_point: np.ndarray
    annual_logit: np.ndarray
    point_count: np.ndarray
    point_mean: np.ndarray
    point_spread: np.ndarray
    covariates: np.ndarray
 
 
def _month_number(months: pd.Series) -> pd.Series:
    """
    Turn month names into month numbers:
 
    Parameters
    ----------
    months : pd.Series.
        Month names such as 'jan' or 'sept', in any case.
 
    Returns
    -------
    result : pd.Series
            The month number from 1 to 12. Raises ValueError naming
            any month that is not in MONTH_NUMBER.
 
    """
    numbers = months.astype(str).str.strip().str.lower().map(MONTH_NUMBER)
    unknown = sorted(months[numbers.isna()].astype(str).unique())
    if unknown:
        raise ValueError(f'unknown month name(s) {unknown}')
    return numbers.astype(int)
 
 
def prepare_year_data(rows: pd.DataFrame) -> YearData:
    """
    Turn one survey year of the joined table into model arrays:
 
    Parameters
    ----------
    rows : pd.DataFrame.
        The rows of one survey round from build_survey_covariate_table,
        one row per child and month.
 
    Returns
    -------
    result : YearData
            The row to child and row to month positions, the point of
            each child, the logit of each child's annual score, the
            count, mean, and spread of those logits at each point,
            and the covariates of each point and month, each
            covariate standardized to mean 0 and standard deviation 1
            over the year.
 
    """
    rows = rows.reset_index(drop=True)
    if rows['malaria_risk_score'].isna().any():
        raise ValueError('malaria_risk_score is missing for some rows')
 
    month_index = _month_number(rows['month']).to_numpy() - 1
    child_index = rows.groupby(CHILD_KEY, sort=False).ngroup().to_numpy()
    point_index = (
        rows.groupby(['latitude', 'longitude'], sort=False)
        .ngroup()
        .to_numpy()
    )
 
    per_child = pd.DataFrame(
        {'point': point_index, 'score': rows['malaria_risk_score']}
    ).groupby(child_index)
    if (per_child['point'].nunique() > 1).any():
        raise ValueError('a child is matched to more than one grid point')
    if (per_child['score'].nunique() > 1).any():
        raise ValueError(
            'malaria_risk_score differs between the months of a child, '
            'but an annual score is expected'
        )
    score = per_child['score'].first().to_numpy()
    annual_logit = logit(np.clip(score, SCORE_CLIP, 1 - SCORE_CLIP))
 
    values = np.full(
        (point_index.max() + 1, 12, len(STATE_COVARIATES)), np.nan
    )
    for k, column in enumerate(STATE_COVARIATES):
        values[point_index, month_index, k] = rows[column].to_numpy(float)
    missing = int(np.isnan(values).sum())
    if missing:
        raise ValueError(
            f'{missing} covariate value(s) are missing, from absent '
            'months or missing readings, and every point needs all 12 '
            'months'
        )
    spread = values.std(axis=(0, 1))
    if (spread == 0).any():
        raise ValueError('a covariate has no variation within the year')
    covariates = (values - values.mean(axis=(0, 1))) / spread
 
    child_point = per_child['point'].first().to_numpy()
    n_points = point_index.max() + 1
    count = np.bincount(child_point, minlength=n_points)
    mean = np.bincount(
        child_point, weights=annual_logit, minlength=n_points
    ) / count
    spread_sum = np.bincount(
        child_point, weights=(annual_logit - mean[child_point]) ** 2,
        minlength=n_points,
    )
    return YearData(
        child_index, month_index, child_point, annual_logit, count, mean,
        spread_sum, covariates,
    )
 
 
def build_state_space_model(data: YearData):
    """
    Build the PyMC state space model of one survey year:
 
    Parameters
    ----------
    data : YearData.
        The arrays from prepare_year_data.
 
    Returns
    -------
    result : pm.Model
            A model that is not yet sampled.
 
    """
    n_points, n_months, n_covariates = data.covariates.shape
    annual_covariates = data.covariates.mean(axis=1)
    several = (data.point_count >= 2) & (data.point_spread > 0)
    identical = int(((data.point_count >= 2) & ~several).sum())
    if identical:
        log.warning(
            '%s location(s) have two or more children with identical '
            'scores and are left out of the spread likelihood',
            identical,
        )
    with pm.Model() as model:
        intercept = pm.Normal(
            'intercept', mu=float(data.annual_logit.mean()), sigma=2.0
        )
        effects = pm.Normal('effects', mu=0.0, sigma=1.0, shape=n_covariates)
        sigma_location = pm.HalfNormal('sigma_location', sigma=1.0)
        phi = pm.Beta('phi', alpha=3.0, beta=2.0)
        sigma_state = pm.HalfNormal('sigma_state', sigma=0.5)
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=1.0)
 
        innovations = pm.Normal(
            'innovations', mu=0.0, sigma=1.0, shape=(n_months, n_points)
        )
        states = [innovations[0] * sigma_state / pt.sqrt(1.0 - phi**2)]
        for month in range(1, n_months):
            states.append(
                phi * states[-1] + sigma_state * innovations[month]
            )
        state = pt.stack(states, axis=1)
        state = state - state.mean(axis=1, keepdims=True)
 
        effect_by_month = pt.dot(data.covariates, effects)
        pm.Deterministic(
            'monthly_deviation',
            effect_by_month
            - effect_by_month.mean(axis=1, keepdims=True)
            + state,
        )
 
        pm.Normal(
            'point_mean',
            mu=intercept + pt.dot(annual_covariates, effects),
            sigma=pt.sqrt(sigma_location**2 + sigma_obs**2 / data.point_count),
            observed=data.point_mean,
        )
        pm.Gamma(
            'point_spread',
            alpha=(data.point_count[several] - 1) / 2,
            beta=1.0 / (2.0 * sigma_obs**2),
            observed=data.point_spread[several],
        )
    return model
 
 
def fit_state_space_model(
    model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int | None = None,
    nuts_sampler: str = 'nutpie',
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Sample a state space model and return its monthly deviations:
 
    Parameters
    ----------
    model : pm.Model.
        The model from build_state_space_model.
    draws : int.
        Posterior draws per chain.
    tune : int.
        Tuning steps per chain.
    chains : int.
        Number of chains.
    cores : int or None.
        Chains run in parallel. None leaves the choice to PyMC.
    nuts_sampler : str.
        Passed to pm.sample, for example 'nutpie' or 'pymc'.
    random_seed : int or None.
        Seed for the sampler.
 
    Returns
    -------
    result : np.ndarray
            The monthly_deviation draws with shape draws, points,
            months, where draws counts all chains together.
 
    """
    with model:
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            nuts_sampler=nuts_sampler, target_accept=0.9,
            random_seed=random_seed, progressbar=False,
        )
 
    divergences = int(trace.sample_stats['diverging'].values.sum())
    summary = az.summary(
        trace,
        var_names=[
            'intercept', 'effects', 'sigma_location', 'phi', 'sigma_state',
            'sigma_obs',
        ],
    )
    r_hat = float(pd.to_numeric(summary['r_hat']).max())
    ess = float(pd.to_numeric(summary['ess_bulk']).min())
    log.info(
        'sampling: %s divergence(s), largest r_hat %.3f, smallest bulk '
        'ESS %.0f', divergences, r_hat, ess,
    )
    if divergences or r_hat > 1.05:
        log.warning('the fit may not have converged, check the sampling')
 
    effects = trace.posterior['effects'].values
    effects = effects.reshape(-1, len(STATE_COVARIATES))
    for name, values in zip(STATE_COVARIATES, effects.T):
        log.info(
            'effect of %s: mean %+.2f, sd %.2f',
            name, values.mean(), values.std(),
        )
 
    deviation = trace.posterior['monthly_deviation'].values
    deviation = deviation.reshape(-1, *deviation.shape[2:])
    log.info(
        'median posterior sd of the monthly deviation: %.2f on the log '
        'odds scale', float(np.median(deviation.std(axis=0))),
    )
    return deviation
 
 
def estimate_monthly_scores(
    data: YearData, deviation: np.ndarray
) -> np.ndarray:
    """
    Turn monthly deviations into a monthly score for each child:
 
    Parameters
    ----------
    data : YearData.
        The arrays from prepare_year_data.
    deviation : np.ndarray.
        The draws returned by fit_state_space_model, shaped draws,
        points, months.
 
    Returns
    -------
    result : np.ndarray
            Shape children, months.
 
    """
    total = np.zeros((len(data.annual_logit), deviation.shape[2]))
    for draw in deviation:
        total += expit(data.annual_logit[:, None] + draw[data.child_point])
    return total / len(deviation)
 
 
def category_edges(scores: pd.Series, categories: pd.Series) -> list[float]:
    """
    Recover the score edges between the four risk categories:
 
    Parameters
    ----------
    scores : pd.Series.
        The annual malaria_risk_score of each row.
    categories : pd.Series.
        The malaria_risk_category built from those scores.
 
    Returns
    -------
    result : list[float]
            The three edges, each halfway between the highest score of
            one category and the lowest score of the next.
 
    """
    grouped = scores.groupby(categories.to_numpy())
    lowest = grouped.min().reindex(RISK_CATEGORIES)
    highest = grouped.max().reindex(RISK_CATEGORIES)
    if lowest.isna().any():
        raise ValueError('the table lacks one of the four risk categories')
 
    edges = []
    for below, above in zip(RISK_CATEGORIES[:-1], RISK_CATEGORIES[1:]):
        if highest[below] >= lowest[above]:
            raise ValueError(f'{below!r} and {above!r} overlap in score')
        edges.append(float((highest[below] + lowest[above]) / 2))
    return edges
 
 
def derive_risk_category(scores: pd.Series, edges: list[float]) -> pd.Series:
    """
    Derive the risk category from a score:
 
    Parameters
    ----------
    scores : pd.Series.
        Scores between 0 and 1.
    edges : list[float].
        The three edges from category_edges.
 
    Returns
    -------
    result : pd.Series
            An ordered category with the labels in RISK_CATEGORIES.
 
    """
    return pd.cut(
        scores, bins=[-np.inf, *edges, np.inf], labels=RISK_CATEGORIES
    )
 
 
def estimate_monthly_risk(
    table: pd.DataFrame,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int | None = None,
    nuts_sampler: str = 'nutpie',
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Replace the annual risk score with a monthly estimate, year by year:
 
    Parameters
    ----------
    table : pd.DataFrame.
        The table from build_survey_covariate_table, holding the
        columns in FINAL_COLUMNS and an annual malaria_risk_score.
    draws : int.
        Posterior draws per chain, for each survey year.
    tune : int.
        Tuning steps per chain.
    chains : int.
        Number of chains.
    cores : int or None.
        Chains run in parallel. None leaves the choice to PyMC.
    nuts_sampler : str.
        Passed to pm.sample.
    random_seed : int or None.
        Seed for the sampler, used for every year.
 
    Returns
    -------
    result : pd.DataFrame
            The same rows and columns as table.
 
    """
    _require_columns(table, FINAL_COLUMNS, 'the survey covariate table')
    table = table.reset_index(drop=True)
    edges = category_edges(
        table['malaria_risk_score'], table['malaria_risk_category']
    )
 
    monthly = pd.Series(np.nan, index=table.index)
    for year, rows in table.groupby('survey_round'):
        log.info(
            'survey year %s: %s child(ren)',
            year, len(rows[CHILD_KEY].drop_duplicates()),
        )
        data = prepare_year_data(rows)
        deviation = fit_state_space_model(
            build_state_space_model(data), draws, tune, chains, cores,
            nuts_sampler, random_seed,
        )
        scores = estimate_monthly_scores(data, deviation)
        monthly.loc[rows.index] = scores[data.child_index, data.month_index]
 
    result = table.copy()
    result['malaria_risk_score'] = monthly
    result['malaria_risk_category'] = derive_risk_category(monthly, edges)
    result = result.rename(columns={'malaria_risk_score': 'monthly_mri'})
    changed = (
        result['malaria_risk_category'].astype(str)
        != table['malaria_risk_category'].astype(str)
    )
    log.info(
        'the category changed in %.1f%% of %s row(s)',
        changed.mean() * 100, len(table),
    )
    return result
 
 
def build_monthly_risk_table(
    input_path: str,
    output_path: str,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int | None = None,
    nuts_sampler: str = 'nutpie',
    random_seed: int | None = None,
    overwrite: bool = False,
) -> str:
    """
    Write the table with monthly risk scores and categories:
 
    Parameters
    ----------
    input_path : str.
        Path of the CSV from build_survey_covariate_table.
    output_path : str.
        Path of the CSV file to write.
    draws : int.
        Passed to estimate_monthly_risk.
    tune : int.
        Passed to estimate_monthly_risk.
    chains : int.
        Passed to estimate_monthly_risk.
    cores : int or None.
        Passed to estimate_monthly_risk.
    nuts_sampler : str.
        Passed to estimate_monthly_risk.
    random_seed : int or None.
        Passed to estimate_monthly_risk.
    overwrite : bool.
        When False and output_path already exists, the existing file
        is left as is and the input is not read again. When True, it
        is regenerated.
 
    Returns
    -------
    result : str
            output_path.
 
    """
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        log.info('%s already present, nothing to do', output_path)
        return output_path
 
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
 
    table = pd.read_csv(input_path, low_memory=False)
    result = estimate_monthly_risk(
        table, draws, tune, chains, cores, nuts_sampler, random_seed
    )
    result.to_csv(output_path, index=False)
    log.info(
        'wrote %s with %s row(s) and %s column(s)',
        output_path, len(result), result.shape[1],
    )
    return output_path


def _calculate_wilson_interval(positive: int,
    total: int, confidence: float = 0.95,) -> tuple[float, float]:
    """Calculate a Wilson confidence interval for a binomial proportion."""

    if total == 0:

        return np.nan, np.nan

    z_score = norm.ppf(1 - (1 - confidence) / 2)
    proportion = positive / total
    denominator = 1 + (z_score**2 / total)
    centre = (proportion + (z_score**2 / (2 * total))) / denominator
    margin = (z_score * np.sqrt((proportion * (1 - proportion) / total)
            + (z_score**2 / (4 * total**2))) / denominator)
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)

    return lower, upper


def validate_mri_categories(
    input_csv: str,
    output_csv: str,
    category_col: str = 'malaria_risk_category',
    rdt_col: str = 'malaria_rdt_result',
    confidence: float = 0.95,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Compare each risk category to the real RDT result, with a Wilson
    confidence interval on each category's positivity rate:
 
    Parameters
    ----------
    input_csv : str.
        Path of a CSV holding category_col and rdt_col, for example
        the output of build_monthly_risk_table.
    output_csv : str.
        Path of the CSV file to write.
    category_col : str.
        Name of the risk category column.
    rdt_col : str.
        Name of the RDT result column. Only 0 (negative) and 1
        (positive) count as a result.
    confidence : float.
        Confidence level of the Wilson interval, for example 0.95
        for a 95% interval.
    overwrite : bool.
        When False and output_csv already exists, it is read back
        and returned instead of being recomputed. When True, it is
        regenerated.
 
    Returns
    -------
    result : pd.DataFrame
            One row per category in RISK_CATEGORIES, in risk order,
            with the count tested, the count and share RDT-positive,
            the Wilson interval on that share, and the risk ratio
            against the lowest risk category. 
 
    """
    output_path = Path(output_csv)
    if output_path.exists() and not overwrite:

        log.info('%s already present, reading it back', output_path)
        return pd.read_csv(output_path)
 
    output_path.parent.mkdir(parents = True, exist_ok = True)
    df = pd.read_csv(input_csv, low_memory =False)
    _require_columns(df, [category_col, rdt_col], input_csv)
 
    rdt = pd.to_numeric(df[rdt_col], errors = 'coerce')
    non_numeric = df[rdt_col].notna() & rdt.isna()
    if non_numeric.any():

        log.info('%s row(s) have a non-numeric %s value and are excluded',
            int(non_numeric.sum()), rdt_col)
 
    non_binary = rdt.notna() & ~rdt.isin([0, 1])
    if non_binary.any():

        codes = sorted(rdt.loc[non_binary].unique().tolist())
        log.info('%s row(s) have a non-binary %s code and are excluded, '
            'the same as a missing result: %s',
            int(non_binary.sum()), rdt_col, codes)
        rdt = rdt.where(~non_binary)
 
    valid = df.loc[rdt.notna()].copy()
    valid['rdt'] = rdt.loc[valid.index].astype(int)
    log.info('%s of %s row(s) have a valid RDT result', len(valid), len(df))
 
    rows = []
    for category in RISK_CATEGORIES:

        in_category = valid.loc[valid[category_col] == category, 'rdt']
        n_total = len(in_category)
        n_positive = int((in_category == 1).sum())
        n_negative = n_total - n_positive
 
        if n_total == 0:

            log.warning('no valid RDT result for category %r', category)
            positivity_pct = np.nan
            ci_lower_pct = np.nan
            ci_upper_pct = np.nan

        else:

            ci_lower, ci_upper = _calculate_wilson_interval(
                n_positive, n_total, confidence)
            positivity_pct = n_positive / n_total * 100
            ci_lower_pct = ci_lower * 100
            ci_upper_pct = ci_upper * 100
 
        rows.append({'risk_category': category, 'n_tested': n_total,
            'rdt_positive': n_positive, 'rdt_negative': n_negative,
            'positivity_pct': positivity_pct, 'ci_lower_pct': ci_lower_pct,
            'ci_upper_pct': ci_upper_pct})
 
    result = pd.DataFrame(rows)
    baseline_pct = result.loc[result['risk_category'] == RISK_CATEGORIES[0], 
                              'positivity_pct'].iloc[0]
    if pd.isna(baseline_pct) or baseline_pct == 0:

        log.warning('the lowest risk category, %r, has no valid RDT result or '
            'zero positives, so risk_ratio_vs_lowest cannot be '
            'computed', RISK_CATEGORIES[0])
        result['risk_ratio_vs_lowest'] = np.nan
    else:

        result['risk_ratio_vs_lowest'] = (result['positivity_pct'] / baseline_pct)
 
    result.to_csv(output_path, index = False)
    log.info('wrote %s with %s row(s)', output_path, len(result))


    return result


def _first_survey_year(label: str) -> int:
    """
    Read the first year out of a survey round label:
 
    Parameters
    ----------
    label : str.
        A round label such as '2009' or '2014-15'.
 
    Returns
    -------
    result : int
            The year before the dash, or the whole label as a year
            when there is no dash, for example 2014 for '2014-15'.
            Raises ValueError when that part is not a 4-digit year.
 
    """
    text = str(label).strip()
    start, _, _ = text.partition('-')
    if len(start) != 4 or not start.isdigit():
        raise ValueError(f'cannot read a survey year from {label!r}')
    return int(start)
 

def split_by_survey_year(
    input_csv: str,
    output_dir: str,
    round_col: str = 'survey_round',
    overwrite: bool = False,
) -> list[str]:
    """
    Split a table into one CSV per survey year:
 
    Parameters
    ----------
    input_csv : str.
        Path of a CSV holding round_col, for example the output of
        build_risk_index.
    output_dir : str.
        Folder the per-year CSV files are written to.
    round_col : str.
        Name of the survey round column, holding labels such as
        '2009' or '2014-15'. The year used is the one before the
        dash, so '2014-15' becomes 2014, not 2015.
    overwrite : bool.
        When False, a per-year file already present in output_dir is
        left as is and not regenerated. When True, it is
        regenerated.
 
    Returns
    -------
    result : list[str]
            Paths of the per-year CSV files present in output_dir
            after the call, one per distinct year found in round_col,
            named <input file name>_<year>.csv. Every row of
            input_csv is written to exactly one file, and the row
            count of each file written this call is logged.
 
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
 
    df = pd.read_csv(input_csv, low_memory=False)
    _require_columns(df, [round_col], input_csv)
 
    year = df[round_col].astype(str).map(_first_survey_year)
 
    stem = Path(input_csv).stem
    written = []
    for survey_year, rows in df.groupby(year):
        out_path = output_path / f'{stem}_{survey_year}.csv'
        if out_path.exists() and not overwrite:
            log.info('%s already present, nothing to do', out_path)
            written.append(str(out_path))
            continue
 
        rows.to_csv(out_path, index=False)
        log.info('wrote %s with %s row(s)', out_path, len(rows))
        written.append(str(out_path))
 
    return written
"""
Role: Attach monthly environmental covariates to the scored survey
table.
Description: Renames each survey round to its latest year, drops rows
with coordinates of 0, 0, and drops the mri_value covariate column
when present. Matches every child to the nearest covariate grid point
among the points that hold the covariate year, then joins that point's
monthly rows. The covariate year is the survey year, or the nearest
year in the covariates when the file lacks it. A child with no point
inside the radius keeps its nearest point, so no row is lost. Writes
one CSV in long form, one row per child and month.
Author: Bonny
"""

import logging
import os
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import expit, logit
from sklearn.neighbors import BallTree
from tqdm import tqdm
from typing import NamedTuple

log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088

# Cluster and household numbers restart each survey round, so a child is
# identified by all four columns together.
CHILD_KEY = ['cluster_number', 'household_number', 'hvidx', 'survey_round']

SURVEY_COLUMNS = [
    'cluster_number', 'household_number', 'hvidx', 'mother_caseid',
    'survey_round', 'LATNUM', 'LONGNUM', 'malaria_risk_score',
    'malaria_risk_category', 'malaria_rdt_result',
]

COVARIATE_COLUMNS = [
    'year', 'longitude', 'latitude', 'ndvi', 'month', 'elevation_m',
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


def _latest_survey_year(label: str) -> int:
    """
    This is a helper function to read the latest year out of a survey round label:

    Parameters
    ----------
    label : str.
        A round label such as '2009' or '2014-15'.

    Returns
    -------
    result : int
            The latest year in the label, for example 2015 for
            '2014-15'. .

    """
    text = str(label).strip()
    start, _, end = text.partition('-')
    valid_start = len(start) == 4 and start.isdigit()
    valid_end = end == '' or (len(end) == 2 and end.isdigit())
    if not (valid_start and valid_end):

        raise ValueError(f'cannot read a survey year from {label!r}')

    year = int(start)
    if end == '':

        return year
    latest = year - year % 100 + int(end)
    if latest < year:

        latest += 100


    return latest


def _pick_covariate_year(
    survey_year: int, available_years: list[int]
) -> int:
    """
    This is a helper function to choose the 
    covariate year to use for a survey year:

    Parameters
    ----------
    survey_year : int.
        Latest year of the survey round.
    available_years : list[int].
        Years present in the covariate file, in ascending order.

    Returns
    -------
    result : int
            survey_year when the covariates hold it..

    """
    if survey_year in available_years:

        return survey_year
    
    return min(available_years, key=lambda year: abs(year - survey_year))


def prepare_data(
    survey: pd.DataFrame, covariates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function cleans the survey and covariate 
    tables before matching:

    Parameters
    ----------
    survey : pd.DataFrame.
        The scored survey table, one row per child.
    covariates : pd.DataFrame.
        The monthly covariate table, one row per point and month.

    Returns
    -------
    result : tuple[pd.DataFrame, pd.DataFrame]
            The survey table and the covariate table. .

    """
    labels = survey['survey_round'].astype(str).unique()
    years = {label: _latest_survey_year(label) for label in labels}
    survey = survey.copy()
    survey['survey_round'] = survey['survey_round'].astype(str).map(years)

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

    available = sorted(int(year) for year in covariates['year'].unique())
    if not available:
        raise ValueError('the covariates hold no year')
    survey['covariate_year'] = survey['survey_round'].map(
        lambda year: _pick_covariate_year(year, available)
    )

    changed = survey[survey['survey_round'] != survey['covariate_year']]
    pairs = changed.groupby(['survey_round', 'covariate_year']).size()
    for (survey_year, used_year), count in pairs.items():
        log.warning(
            '%s row(s) of survey year %s use covariate year %s, the '
            'nearest year in the covariates',
            count, survey_year, used_year,
        )
    return survey, covariates


def find_nearest_points(
    query: pd.DataFrame,
    candidates: pd.DataFrame,
    radius_km: float = 5.0,
) -> pd.DataFrame:
    """
    This function finds the nearest candidate 
    point for every query point:

    Parameters
    ----------
    query : pd.DataFrame.
        Columns latitude and longitude in degrees, one row per query
        point.
    candidates : pd.DataFrame.
        Columns latitude and longitude in degrees, one row per
        candidate point. Positions are counted from the first row.
    radius_km : float.
        Distance in kilometres used only to count and report the
        query points that have no candidate inside it.

    Returns
    -------
    result : pd.DataFrame
            One row per query point, with the index of query, and the
            columns nearest_latitude, nearest_longitude, and
            distance_km.

    """
    for name, frame in (('query', query), ('candidates', candidates)):
        if frame[['latitude', 'longitude']].isna().any().any():
            raise ValueError(f'{name} has a missing latitude or longitude')
    if candidates.empty:
        raise ValueError('there are no candidate points to match against')

    tree = BallTree(
        np.radians(candidates[['latitude', 'longitude']].to_numpy()),
        metric='haversine',
    )
    distance, position = tree.query(
        np.radians(query[['latitude', 'longitude']].to_numpy()), k=1
    )
    distance_km = distance[:, 0] * EARTH_RADIUS_KM
    nearest = candidates.iloc[position[:, 0]]

    beyond = distance_km > radius_km
    if beyond.any():
        log.warning(
            '%s of %s point(s) have no candidate within %s km and keep '
            'their nearest one, up to %.1f km away',
            int(beyond.sum()), len(query), radius_km, distance_km.max(),
        )
    return pd.DataFrame(
        {
            'nearest_latitude': nearest['latitude'].to_numpy(),
            'nearest_longitude': nearest['longitude'].to_numpy(),
            'distance_km': distance_km,
        },
        index=query.index,
    )


def merge_covariates(
    survey: pd.DataFrame,
    covariates: pd.DataFrame,
    radius_km: float = 5.0,
) -> pd.DataFrame:
    """
    Join each child to the monthly covariates of its nearest point:

    Parameters
    ----------
    survey : pd.DataFrame.
        The survey table returned by prepare_data, with a
        covariate_year column.
    covariates : pd.DataFrame.
        The covariate table returned by prepare_data.
    radius_km : float.
        Passed to find_nearest_points.

    Returns
    -------
    result : pd.DataFrame
            One row per child and covariate month.

    """
    _require_columns(survey, ['covariate_year'], 'the survey table')

    matched = []
    for year, children in survey.groupby('covariate_year'):
        in_year = covariates[covariates['year'] == year]
        candidates = (
            in_year[['latitude', 'longitude']]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        log.info(
            'covariate year %s: matching %s row(s) to %s point(s)',
            year, len(children), len(candidates),
        )
        query = children[['LATNUM', 'LONGNUM']].rename(
            columns={'LATNUM': 'latitude', 'LONGNUM': 'longitude'}
        )
        nearest = find_nearest_points(query, candidates, radius_km)
        matched.append(children.join(nearest))
    children = pd.concat(matched)

    # The matched coordinates are copied from the covariate table, so the
    # float keys are identical and the join needs no tolerance.
    merged = children.merge(
        covariates,
        left_on=['covariate_year', 'nearest_latitude', 'nearest_longitude'],
        right_on=['year', 'latitude', 'longitude'],
        how='left',
    )
    merged['month_num'] = _month_number(merged['month'])
    unmatched = int(merged['year'].isna().sum())
    if unmatched:
        raise ValueError(f'{unmatched} row(s) found no covariate row')

    log.info(
        '%s child row(s) matched, distance to the point: median %.1f '
        'km, maximum %.1f km',
        len(children), children['distance_km'].median(),
        children['distance_km'].max(),
    )
    return merged


def build_survey_covariate_table(
    survey_path: str,
    covariate_path: str,
    output_path: str,
    radius_km: float = 5.0,
    overwrite: bool = False,
) -> str:
    """
    Write the survey table joined to its monthly covariates:
 
    Parameters
    ----------
    survey_path : str.
        Path of the scored survey CSV, for example the output of
        merge_risk_with_outcome.
    covariate_path : str.
        Path of the monthly covariate CSV.
    output_path : str.
        Path of the CSV file to write.
    radius_km : float.
        Passed to find_nearest_points.
    overwrite : bool.
        When False and output_path already exists, the existing file
        is left as is and the inputs are not read again. When True,
        it is regenerated.
 
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
 
    survey = pd.read_csv(survey_path, low_memory=False)
    covariates = pd.read_csv(covariate_path, low_memory=False)
    _require_columns(survey, SURVEY_COLUMNS, survey_path)
    _require_columns(covariates, COVARIATE_COLUMNS, covariate_path)
 
    survey, covariates = prepare_data(survey, covariates)
    merged = merge_covariates(survey, covariates, radius_km)
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
    several = data.point_count >= 2
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
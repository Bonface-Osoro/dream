import configparser
import logging
import os
import time
from dream.surv_index import *
from dream.survGeo import *

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
 
CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.join(os.path.dirname(__file__), "script_config.ini"))
BASE_PATH = CONFIG["file_locations"]["base_path"]
 
DATA_RAW = os.path.join(BASE_PATH, 'raw', 'dhs')
OUTPUT_PATH = os.path.join(BASE_PATH, '..', 'results', 'processed', 'UGA_dhs', 'file1_UGA_malaria_survey_input.csv')
GPS_RAW = os.path.join(BASE_PATH, 'raw', 'gps_surv')
MERGED = os.path.join(BASE_PATH, '..', 'results', 'processed', 'UGA_dhs', 'file2_UGA_malaria_survey_with_gps.csv')
REFINED = os.path.join(BASE_PATH, '..', 'results', 'processed', 'UGA_dhs', 'file3_UGA_malaria_survey_model_input.csv')
RISK_INDEXED = os.path.join(BASE_PATH, '..', 'results', 'processed', 'UGA_dhs', 'file4_UGA_malaria_risk_index.csv')
VALIDATION = os.path.join(BASE_PATH, '..', 'results', 'processed', 'UGA_dhs', 'file5_UGA_malaria_validation.csv')
COVARIATES = os.path.join(BASE_PATH, '..', 'results', 'final', 'ecological_predictors_with_mri.csv')
SURV_COVARIATES = os.path.join(BASE_PATH, '..', 'results', 'processed', 'UGA_dhs', 'file6_UGA_malaria_annual_risk_covariates.csv')
MONTHLY_COVARIATES = os.path.join(BASE_PATH, '..', 'results', 'processed', 'UGA_dhs', 'file7_UGA_malaria_monthly_risk_covariates.csv')

start = time.time()
result_path = build_malaria_index_table(base_dir = DATA_RAW, 
                                        output_path = OUTPUT_PATH)
merge_with_gps_clusters(OUTPUT_PATH, GPS_RAW, MERGED)
select_model_columns(MERGED, REFINED)
build_risk_index(REFINED, RISK_INDEXED)
merge_risk_with_outcome(RISK_INDEXED, REFINED, VALIDATION)
build_survey_covariate_table(VALIDATION, COVARIATES, SURV_COVARIATES)
build_monthly_risk_table(SURV_COVARIATES, MONTHLY_COVARIATES)
elapsed = time.time() - start
 
log.info("wrote %s in %.1fs", result_path, elapsed) 
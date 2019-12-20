import os

TEST_SUFFIX = os.getenv("TEST_SUFFIX", "") # takes value specified in pytest.ini file when launching tests.

PROJECT_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",)
RAW_DATA = os.path.join(PROJECT_ROOT_PATH, "data", "raw")
INTERIM_DATA = os.path.join(PROJECT_ROOT_PATH, "data", "interim")
OUTPUT_DATA = os.path.join(PROJECT_ROOT_PATH, "data", "output")

ENERGY_CONSUMPTION = os.path.join(RAW_DATA, "eco2mix-regional-cons-def.csv")
MAIN_METEO = os.path.join(RAW_DATA, "full_meteo.csv")
LAST_METEO_PARIS = os.path.join(RAW_DATA, "meteo_paris_2019_juin_sept.csv")

FULL_METEO = os.path.join(INTERIM_DATA, "full_meteo.csv")
REGION_DF_DICT = os.path.join(INTERIM_DATA, "region_df_dict{}.pkl".format(TEST_SUFFIX))

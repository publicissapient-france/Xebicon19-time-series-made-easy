import os
import logging
from datetime import datetime


def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


TEST_SUFFIX = os.getenv("TEST_SUFFIX", "") # takes value specified in pytest.ini file when launching tests.

PROJECT_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",)
# data path will be /data for project and /data_test for tests
DATA_PATH = create_folder(os.path.join(PROJECT_ROOT_PATH, f"data{TEST_SUFFIX}"))
LOG_PATH = create_folder(os.path.join(DATA_PATH, "logs"))

today_str = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_PATH, today_str + ".log")
logging_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=logging_format)

# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter(logging_format))
logging.getLogger('').addHandler(console)

RAW_DATA = create_folder(os.path.join(DATA_PATH, "raw"))
INTERIM_DATA = create_folder(os.path.join(DATA_PATH, "interim"))
OUTPUT_DATA = create_folder(os.path.join(DATA_PATH, "output"))
MODELS = create_folder(os.path.join(DATA_PATH, "models"))

PLOTS = create_folder(os.path.join(OUTPUT_DATA, "plots"))

ENERGY_CONSUMPTION_URL = "https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-regional-cons-def/download/?format=csv&timezone=Europe/Paris&use_labels_for_header=true&csv_separator=%3B"

ENERGY_CONSUMPTION_FILE = "eco2mix-regional-cons-def.csv"
ENERGY_CONSUMPTION = os.path.join(RAW_DATA, ENERGY_CONSUMPTION_FILE)
MAIN_METEO_FILE = "full_meteo.csv"
MAIN_METEO = os.path.join(RAW_DATA, MAIN_METEO_FILE)
LAST_METEO_FILE = "meteo_paris_2019_juin_sept.csv"
LAST_METEO_PARIS = os.path.join(RAW_DATA, LAST_METEO_FILE)

FULL_METEO = os.path.join(INTERIM_DATA, "full_meteo.csv")
REGION_DF_DICT = os.path.join(INTERIM_DATA, "region_df_dict.pkl")

PROPHET_2_YEARS_MODEL = "prophet_2_years.pkl"
PROPHET_2_YEARS_WEATHER_MODEL = "prophet_2_years_with_weather.pkl"

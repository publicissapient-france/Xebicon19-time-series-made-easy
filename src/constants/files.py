import os
import logging
from datetime import datetime


def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


PROJECT_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",)
LOG_PATH = create_folder(os.path.join(PROJECT_ROOT_PATH, "data", "logs"))

today_str = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_PATH, today_str + ".log")
logging_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=logging_format)

# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter(logging_format))
logging.getLogger('').addHandler(console)

TEST_SUFFIX = os.getenv("TEST_SUFFIX", "") # takes value specified in pytest.ini file when launching tests.

RAW_DATA = os.path.join(PROJECT_ROOT_PATH, "data", "raw")
INTERIM_DATA = create_folder(os.path.join(PROJECT_ROOT_PATH, "data", "interim"))
OUTPUT_DATA = create_folder(os.path.join(PROJECT_ROOT_PATH, "data", "output"))

PLOTS = create_folder(os.path.join(OUTPUT_DATA, "plots{}".format(TEST_SUFFIX)))

ENERGY_CONSUMPTION_URL = url = "https://opendata.reseaux-energies.fr/explore/dataset/eco2mix-regional-cons-def/download/?format=csv&timezone=Europe/Paris&use_labels_for_header=true&csv_separator=%3B"
ENERGY_CONSUMPTION = os.path.join(RAW_DATA, "eco2mix-regional-cons-def.csv")
MAIN_METEO = os.path.join(RAW_DATA, "full_meteo.csv")
LAST_METEO_PARIS = os.path.join(RAW_DATA, "meteo_paris_2019_juin_sept.csv")

FULL_METEO = os.path.join(INTERIM_DATA, "full_meteo.csv")
REGION_DF_DICT = os.path.join(INTERIM_DATA, "region_df_dict.pkl")

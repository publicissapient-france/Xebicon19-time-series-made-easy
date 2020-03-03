import pandas as pd
import os

import src.constants.files as files
import src.constants.columns as c
import src.constants.models as md

from datetime import datetime, timedelta

import logging


REAL_RAW_DATA_PATH = os.path.join(files.PROJECT_ROOT_PATH, "data", "raw")
TEST_RAW_DATA_PATH = os.path.join(files.PROJECT_ROOT_PATH, "data_test", "raw")

file_date_dict = {
    files.ENERGY_CONSUMPTION_FILE: c.EnergyConso.TIMESTAMP,
    files.MAIN_METEO_FILE: c.Meteo.DATE,
    files.LAST_METEO_FILE: c.Meteo.DATE
}


def prepare_raw_test_data(force_recompute=False):
    real_raw_csv_files = [file for file in os.listdir(REAL_RAW_DATA_PATH) if file.endswith(".csv")]
    test_raw_csv_files = [file for file in os.listdir(TEST_RAW_DATA_PATH) if file.endswith(".csv")]

    files_to_copy_in_test = [file for file in real_raw_csv_files if file not in test_raw_csv_files]

    if force_recompute:
        files_to_copy_in_test = real_raw_csv_files

    for file in files_to_copy_in_test:
        logging.info(f"Truncating {file} and writing truncated version into test raw data folder.")
        df_date = file_date_dict[file]
        df = pd.read_csv(os.path.join(REAL_RAW_DATA_PATH, file))

        # Keep one month before end train date and nb pred hours after.
        df_trunc = df[
            (df[df_date].apply(lambda x: str(x)[:10])
                >= datetime.strftime(md.END_TRAIN_DATE - timedelta(days=30), "%Y-%m-%d"))
            & (df[df_date].apply(lambda x: str(x)[:10])
                <= datetime.strftime(md.END_TRAIN_DATE + timedelta(hours=md.NB_HOURS_PRED), "%Y-%m-%d"))
        ]

        df_trunc.to_csv(os.path.join(TEST_RAW_DATA_PATH, file))

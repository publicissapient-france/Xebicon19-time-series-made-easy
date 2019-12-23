import pickle

import src.constants.files as files
import src.constants.models as md

from src.deepar.deepar_core import train_predictor

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_to_compare_3_ways(max_epochs, learning_rate):
    region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))

    logging.info("Training model for IDF alone with {} epochs and lr of {}".format(max_epochs, learning_rate))
    idf_list = ["Ile-de-France"]
    train_predictor(region_df_dict, md.END_TRAIN_DATE, idf_list, max_epochs, learning_rate,
                    "Consommation (MW)", feat_dynamic_cols=None)

    logging.info("Training model for all_regions with {} epochs and lr of {}".format(max_epochs, learning_rate))
    all_regions = list(region_df_dict.keys())
    train_predictor(region_df_dict, md.END_TRAIN_DATE, all_regions, max_epochs, learning_rate,
                    "Consommation (MW)", feat_dynamic_cols=None)

    logging.info("Training model for IDF alone with {} epochs and lr of {} and temperature as covariate"
                 .format(max_epochs, learning_rate))
    idf_list = ["Ile-de-France"]
    train_predictor(region_df_dict, md.END_TRAIN_DATE, idf_list, max_epochs, learning_rate,
                    "Consommation (MW)", feat_dynamic_cols=["max_temp_paris"])
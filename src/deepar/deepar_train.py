import pickle

import src.constants.files as files
import src.constants.models as md
import src.constants.columns as c
from src.deepar.deepar_core import predictor_path, train_predictor, DEEPAR_MODELS_PATH

import logging
import os


def deepar_training_confs(region_df_dict):
    return [{"region_list": [md.IDF], "feat_dynamic_cols": None},
            {"region_list": list(region_df_dict.keys()), "feat_dynamic_cols": None},
            {"region_list": [md.IDF], "feat_dynamic_cols": [c.Meteo.MAX_TEMP_PARIS]}]


def train_to_compare_3_ways(max_epochs, learning_rate):
    with open(files.REGION_DF_DICT, "rb") as f:
        region_df_dict = pickle.load(f)
    training_confs = deepar_training_confs(region_df_dict)

    for conf in training_confs:
        regions_list = conf["region_list"]
        feat_dynamic_cols = conf["feat_dynamic_cols"]
        deepar_training_logging(region_df_dict, regions_list, feat_dynamic_cols, max_epochs, learning_rate)
        train_predictor(region_df_dict, md.END_TRAIN_DATE, regions_list, max_epochs, learning_rate,
                        c.EnergyConso.CONSUMPTION, feat_dynamic_cols=feat_dynamic_cols)


def deepar_training_logging(region_df_dict, region_list, feat_dynamic_cols, max_epochs, learning_rate):
    if len(region_list) == 1:
        region_str = f"{region_list[0]} alone"
    elif len(region_list) == len(list(region_df_dict.keys())):
        region_str = "all regions"
    else:
        region_str = f"{len(region_list)} regions"

    if feat_dynamic_cols is not None:
        cols_str = ", ".join(feat_dynamic_cols)
        cov_str = f" and {cols_str} as covariate(s)"
    else:
        cov_str = ""

    logging.info(f"Training model for {region_str} with {max_epochs} epochs and lr of {learning_rate}{cov_str}.")


def train_idf_n_times(max_epochs, learning_rate, n_trainings):
    with open(files.REGION_DF_DICT, "rb") as f:
        region_df_dict = pickle.load(f)

    regions_list = [md.IDF]
    feat_dynamic_cols = None

    nth_model_training_path = predictor_path(
        region_df_dict, regions_list, max_epochs, learning_rate, feat_dynamic_cols, trial_number=n_trainings)
    model_dir, nth_model_name = os.path.split(nth_model_training_path)

    while True:
        if nth_model_name in os.listdir(DEEPAR_MODELS_PATH):
            break
        train_predictor(region_df_dict, md.END_TRAIN_DATE, regions_list, max_epochs, learning_rate,
                        c.EnergyConso.CONSUMPTION, feat_dynamic_cols=feat_dynamic_cols)


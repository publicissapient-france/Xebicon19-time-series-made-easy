from src.preprocess import preprocess_meteo_data, preprocess_energy_consumption_data
from src.evaluation.plots import plot_consumptions
from src.deepar.deepar_train import train_to_compare_3_ways
from src.prophet.prophet_train import prophet_train
from src.evaluation.evaluation import evaluate_models

import src.constants.models as md

import os


def main(bool_dict):
    if bool_dict["preprocess_data"]:
        preprocess_meteo_data()
        region_df_dict = preprocess_energy_consumption_data()
        plot_consumptions(region_df_dict, 2018, 12)

    if bool_dict["train_deepar"]:
        train_to_compare_3_ways(os.getenv("TEST_MAX_EPOCHS", md.MAX_EPOCHS), md.LEARNING_RATE)

    if bool_dict["train_prophet"]:
        prophet_train()

    if bool_dict["evaluate"]:
        evaluate_models()



if __name__ == "__main__":
    bool_dict = {"preprocess_data": False,
                 "train_deepar": False,
                 "train_prophet": False,
                 "evaluate": True}
    main(bool_dict)

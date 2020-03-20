from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from src.preprocess import preprocess_meteo_data, preprocess_energy_consumption_data
from src.evaluation.plots import plot_consumptions
from src.deepar.deepar_train import train_to_compare_3_ways, train_idf_n_times
from src.prophet.prophet_train import prophet_train
from src.evaluation.evaluation import evaluate_models
from src.evaluation.deepar_stability_study import (
    run_model_stability_study, plot_model_stability_study_results, run_num_eval_samples_stability_study,
    plot_num_eval_samples_study_results)
from src.sarima.sarima_train import sarima_train

import src.constants.models as md
import src.constants.files as files

import os
import logging


TDV = {
    "max_epochs": os.getenv("TEST_MAX_EPOCH", md.MAX_EPOCHS),
    "max_epochs_list": eval(os.getenv("TEST_MAX_EPOCH_LIST", md.DEEPAR_MAX_EPOCH_LIST_STR)),
    "max_nb_trainings": eval(os.getenv("MAX_NB_TRAININGS", md.MAX_NB_TRAININGS)),
    "max_arima_param_range": int(os.getenv("MAX_ARIMA_PARAM_RANGE", 3)),
}


def main(bool_dict, tdv):
    """
    Run all steps of the project.

    :param bool_dict: boolean dictionary to select the steps to run.
    :param tdv: test dependent variable. The values taken from this dictionary will come from pytest.ini when tests are
    launched with pytest, or take their normal value when the main function is launched directly. This allows to have
    a "free" integration test on a small sample of the raw data. The tdv dict is set as input to the main function to
    have a clear view of test dependent variables and decouple test from production code as much as possible while
    keeping this "free" integration test.
    :return:
    """
    logging.info("Launching main.")

    if bool_dict["preprocess_data"]:
        preprocess_meteo_data()
        region_df_dict = preprocess_energy_consumption_data()
        plot_consumptions(region_df_dict, 2018, 12)

    if bool_dict["train_deepar"]:
        train_to_compare_3_ways(tdv["max_epochs"], md.LEARNING_RATE)

    if bool_dict["train_prophet"]:
        prophet_train()

    if bool_dict["multiple_deepar_trainings"]:
        logging.info("Training deepar multiple times to test stability.")
        for max_epochs in tdv["max_epochs_list"]:
            train_idf_n_times(max_epochs, md.LEARNING_RATE, n_trainings=tdv["max_nb_trainings"])

    if bool_dict["run_deepar_stability_study"]:
        run_model_stability_study(tdv["max_epochs_list"], tdv["max_nb_trainings"])
        plot_model_stability_study_results(tdv["max_epochs_list"], tdv["max_nb_trainings"])

        run_num_eval_samples_stability_study(tdv["max_epochs_list"][0], trial_nb=1)
        plot_num_eval_samples_study_results(tdv["max_epochs_list"][0], trial_nb=1)

    if bool_dict["run_arima_training"]:
        sarima_train(tdv["max_arima_param_range"])

    if bool_dict["evaluate"]:
        evaluate_models(tdv["max_epochs"])


if __name__ == "__main__":
    bool_dict = {"preprocess_data": False,
                 "train_deepar": False,
                 "train_prophet": False,
                 "multiple_deepar_trainings": False,
                 "run_deepar_stability_study": False,
                 "run_arima_training": False,
                 "evaluate": True}
    main(bool_dict, TDV)

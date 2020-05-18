from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from src.preprocess import preprocess_meteo_data, preprocess_energy_consumption_data
from src.evaluation.plots import plot_consumptions
from src.deepar.deepar_train import train_to_compare_3_ways, train_idf_n_times
from src.prophet.prophet_train import prophet_train
from src.evaluation.evaluation import evaluate_models
from src.evaluation.deepar_stability_study import (
    run_model_stability_study, plot_model_stability_study_results, run_num_eval_samples_stability_study,
    plot_num_eval_samples_study_results, free_vs_fixed_seeds_plot, plot_model_performance_on_multiple_dates,
    plot_retraining_free_seeds_vs_multiple_dates)
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
    "nb_pred_num_eval_samples_study": int(
        os.getenv("TEST_NB_PRED_NUM_EVAL_SAMPLES_STUDY", md.NB_PRED_NUM_EVAL_SAMPLES_STUDY))
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
        logging.info("Training deepar multiple times for stability tests.")
        for max_epochs in tdv["max_epochs_list"]:
            logging.info(f"Training with {max_epochs} epochs")
            train_idf_n_times(max_epochs, md.LEARNING_RATE, n_trainings=tdv["max_nb_trainings"])
        # Sanity check to make sure that setting the seeds results in stable results
        logging.info(f"Training with fixed_seeds")
        train_idf_n_times(
            tdv["max_epochs_list"][0], md.LEARNING_RATE, n_trainings=tdv["max_nb_trainings"], fixed_seeds=True)

    if bool_dict["run_deepar_stability_study"]:
        logging.info("Running model stability study with free random seeds.")
        run_model_stability_study(tdv["max_epochs_list"], tdv["max_nb_trainings"])
        plot_model_stability_study_results(tdv["max_epochs_list"], tdv["max_nb_trainings"])

        logging.info("Running model stability study with fixed random seeds.")
        run_model_stability_study([tdv["max_epochs_list"][0]], tdv["max_nb_trainings"], fixed_seeds=True)
        free_vs_fixed_seeds_plot()

        logging.info("Running num_eval_samples stability study.")
        run_num_eval_samples_stability_study(
         tdv["max_epochs_list"][0], trial_nb=1, nb_pred=tdv["nb_pred_num_eval_samples_study"],
         num_eval_samples_list=[10, 100, 200],
         results_file=files.NUM_EVAL_SAMPLES_STUDY_FILE)
        plot_num_eval_samples_study_results(tdv["max_epochs_list"][0], trial_nb=1, prediction_date=md.END_TRAIN_DATE)

        logging.info("Evaluating performance over 6 months at 2 week intervals"
                     " with same model and num_eval_samples of 100.")
        run_num_eval_samples_stability_study(
         tdv["max_epochs_list"][0], trial_nb=1, nb_pred=1, num_eval_samples_list=[100],
         results_file=files.MULTIPLE_DAYS_PERFORMANCE_STUDY_FILE, nb_additional_pred_days=12)
        plot_model_performance_on_multiple_dates()
        plot_retraining_free_seeds_vs_multiple_dates()

    if bool_dict["run_arima_training"]:
        sarima_train(tdv["max_arima_param_range"])

    if bool_dict["evaluate"]:
        evaluate_models(tdv["max_epochs"])


if __name__ == "__main__":
    bool_dict = {"preprocess_data": False,
                 "train_deepar": False,
                 "train_prophet": False,
                 "multiple_deepar_trainings": False,
                 "run_deepar_stability_study": True,
                 "run_arima_training": False,
                 "evaluate": False}
    main(bool_dict, TDV)

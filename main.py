from src.preprocess import preprocess_meteo_data, preprocess_energy_consumption_data
from src.evaluation.plots import plot_consumptions
from src.deepar.deepar_train import train_to_compare_3_ways, train_idf_n_times
from src.prophet.prophet_train import prophet_train
from src.evaluation.evaluation import evaluate_models
from src.evaluation.deepar_stability_study import (
    run_model_stability_study, plot_model_stability_study_results, run_num_eval_samples_stability_study,
    plot_num_eval_samples_study_results)

import src.constants.models as md

import os
import logging


def main(bool_dict):
    if bool_dict["preprocess_data"]:
        preprocess_meteo_data()
        region_df_dict = preprocess_energy_consumption_data()
        plot_consumptions(region_df_dict, 2018, 12)

    if bool_dict["train_deepar"]:
        train_to_compare_3_ways(os.getenv("TEST_MAX_EPOCH", md.MAX_EPOCH), md.LEARNING_RATE)

    if bool_dict["train_prophet"]:
        prophet_train()

    if bool_dict["evaluate"]:
        evaluate_models()

    if bool_dict["multiple_deepar_trainings"]:
        logging.info("Training deepar multiple times to test stability.")
        for max_epochs in eval(os.getenv("TEST_MAX_EPOCH_LIST", md.DEEPAR_MAX_EPOCH_LIST_STR)):
            train_idf_n_times(max_epochs, md.LEARNING_RATE,
                              n_trainings=eval(os.getenv("MAX_NB_TRAININGS", md.MAX_NB_TRAININGS)))

    if bool_dict["run_deepar_stability_study"]:
        max_epoch_list = eval(os.getenv("TEST_MAX_EPOCH_LIST", md.DEEPAR_MAX_EPOCH_LIST_STR))
        nb_trials = eval(os.getenv("MAX_NB_TRAININGS", md.MAX_NB_TRAININGS))
        run_model_stability_study(max_epoch_list, nb_trials)
        plot_model_stability_study_results(max_epoch_list, nb_trials)

        run_num_eval_samples_stability_study(max_epoch_list[0], trial_nb=1)
        plot_num_eval_samples_study_results(max_epoch_list[0], trial_nb=1)


if __name__ == "__main__":
    bool_dict = {"preprocess_data": False,
                 "train_deepar": False,
                 "train_prophet": False,
                 "evaluate": False,
                 "multiple_deepar_trainings": False,
                 "run_deepar_stability_study": True}
    main(bool_dict)

import os
import pickle
import logging
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path

import src.constants.files as files
import src.constants.models as md

from src.evaluation.evaluation import prepare_data_for_deepar_plot
from src.evaluation.deepar_plots import plot_forecasts

STABILITY_STUDY_PATH = files.create_folder(
    os.path.join(files.OUTPUT_DATA, f"deepar_stability_study{files.TEST_SUFFIX}"))

STABILITY_STUDY_PLOTS = files.create_folder(os.path.join(STABILITY_STUDY_PATH, "plots"))

DEEPAR_STABILITY_STUDY_RESULTS = os.path.join(STABILITY_STUDY_PATH, "stability_study_results.csv")


def run_model_stability_study(max_epoch_list, nb_trials):
    """
    Run evaluation of same models trained nb_trials times with different max_epoch to assess deepar stability.

    :param max_epoch_list: list of max_epoch used in the various models trained.
    :param nb_trials: number of times that each model with given max_epoch has been trained.
    :return: Saves dataframe with results and plots of predictions for each model.
    """
    region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
    stability_study_results = []

    for max_epoch in max_epoch_list:
        logging.info(f"Running prediction 10 times with model trained with {max_epoch} epochs.")
        for trial_nb in range(1, nb_trials + 1):
            forecasts, tss, model_pkl_path = prepare_data_for_deepar_plot(
                region_df_dict, [md.IDF], None, max_epoch, md.LEARNING_RATE, trial_nb)

            fig_path = os.path.join(STABILITY_STUDY_PLOTS, f"{Path(model_pkl_path).name}.png")
            mape = plot_forecasts(
                region_df_dict, md.END_TRAIN_DATE, tss, forecasts, past_length=2 * md.NB_HOURS_PRED, fig_path=fig_path)
            result_dict = {"learning_rate": md.LEARNING_RATE,
                           "max_epoch": max_epoch,
                           "trial_nb": trial_nb,
                           "prediction_date": md.END_TRAIN_DATE,
                           "nb_days_pred": int(md.NB_HOURS_PRED / 24),
                           "MAPE": mape}

            stability_study_results.append(result_dict)

    stability_study_results_df = pd.DataFrame.from_records(stability_study_results)

    stability_study_results_df.to_csv(DEEPAR_STABILITY_STUDY_RESULTS, index=False)


def plot_model_stability_study_results(max_epoch_list, nb_trials):
    results = pd.read_csv(DEEPAR_STABILITY_STUDY_RESULTS)
    results = results[results["max_epoch"].isin(max_epoch_list) & (results["trial_nb"] <= nb_trials)]
    # Turn MAPE into percentage
    results["MAPE"] = results["MAPE"]*100

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(1, figsize=(20, 8))
    ax = sns.boxplot(x="max_epoch", y="MAPE", data=results, whis=[10, 90])
    ax = sns.swarmplot(x="max_epoch", y="MAPE", data=results, color=".25", s=10)
    plt.xlabel("Number of training epochs")
    plt.ylabel("MAPE (%)")
    plt.ylim([0, 25])
    plt.title(f"Distribution of MAPE on January 1st prediction for {nb_trials} trainings")
    plt.savefig(os.path.join(STABILITY_STUDY_PATH, "max_epoch_stability_boxplot.png"))
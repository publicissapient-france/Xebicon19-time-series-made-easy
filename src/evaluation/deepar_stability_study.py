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

MODEL_STABILITY_STUDY_PLOTS = files.create_folder(os.path.join(STABILITY_STUDY_PATH, "model_stability_study_plots"))

MODEL_STABILITY_STUDY_RESULTS = os.path.join(STABILITY_STUDY_PATH, "model_stability_study_results.csv")

NUM_EVAL_SAMPLES_STABILITY_STUDY_PLOTS = files.create_folder(
    os.path.join(STABILITY_STUDY_PATH, "num_eval_samples_stability_study_plots"))

NUM_EVAL_SAMPLES_STABILITY_STUDY_RESULTS = os.path.join(
    STABILITY_STUDY_PATH, "num_eval_samples_stability_study_results.csv")


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

            fig_path = os.path.join(MODEL_STABILITY_STUDY_PLOTS, f"{Path(model_pkl_path).name}.png")
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

    stability_study_results_df.to_csv(MODEL_STABILITY_STUDY_RESULTS, index=False)


def run_num_eval_samples_stability_study(max_epoch, trial_nb):
    region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
    stability_study_results = []

    num_eval_samples_list = [10, 100, 1000]
    logging.info(
        f"Running prediction with model trained with {max_epoch} epochs"
        f" with sum eval samples in {num_eval_samples_list}.")
    for num_eval_samples in num_eval_samples_list:

        forecasts, tss, model_pkl_path = prepare_data_for_deepar_plot(
            region_df_dict, [md.IDF], None, max_epoch, md.LEARNING_RATE, trial_nb, num_eval_samples=num_eval_samples)

        fig_path = os.path.join(
            NUM_EVAL_SAMPLES_STABILITY_STUDY_PLOTS, f"{Path(model_pkl_path).name}_{num_eval_samples}_samples.png")
        mape = plot_forecasts(
            region_df_dict, md.END_TRAIN_DATE, tss, forecasts, past_length=2 * md.NB_HOURS_PRED, fig_path=fig_path)
        result_dict = {"learning_rate": md.LEARNING_RATE,
                       "max_epoch": max_epoch,
                       "trial_nb": trial_nb,
                       "prediction_date": md.END_TRAIN_DATE,
                       "nb_days_pred": int(md.NB_HOURS_PRED / 24),
                       "num_eval_samples": num_eval_samples,
                       "MAPE": mape}

        stability_study_results.append(result_dict)

    stability_study_results_df = pd.DataFrame.from_records(stability_study_results)

    stability_study_results_df.to_csv(NUM_EVAL_SAMPLES_STABILITY_STUDY_RESULTS, index=False)


def plot_model_stability_study_results(max_epoch_list, nb_trials):
    results = pd.read_csv(MODEL_STABILITY_STUDY_RESULTS)
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
    plt.savefig(os.path.join(STABILITY_STUDY_PATH, "model_stability_boxplot.png"))
    
    
def plot_num_eval_samples_study_results(max_epoch, trial_nb):
    results = pd.read_csv(NUM_EVAL_SAMPLES_STABILITY_STUDY_RESULTS)
    results = results[(results["max_epoch"]==max_epoch) & (results["trial_nb"]==trial_nb)]

    # Turn MAPE into percentage
    results["MAPE_r"] = results["MAPE"].apply(lambda x: round(100*x, 2))

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(1, figsize=(20, 8))
    ax = sns.boxplot(x="num_eval_samples", y="MAPE_r", data=results, whis=[10, 90])
    ax = sns.swarmplot(x="num_eval_samples", y="MAPE_r", data=results, color=".25", s=10)
    plt.xlabel("Number of samples for prediction")
    plt.ylabel("MAPE (%)")
    plt.ylim([5, 8])
    plt.title("Distribution of MAPE on January 1st prediction for different eval sample sizes")
    plt.savefig(os.path.join(STABILITY_STUDY_PATH, "num_eval_samples_stability_boxplot.png"))

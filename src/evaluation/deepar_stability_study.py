import os
import pickle
import logging
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from pathlib import Path

import src.constants.files as files
import src.constants.models as md
import src.constants.columns as c

from src.evaluation.evaluation import prepare_data_for_deepar_plot
from src.evaluation.plots import plot_deepar_forecasts
from src.deepar.deepar_core import predictor_path, make_predictions

STABILITY_STUDY_PATH = files.create_folder(
    os.path.join(files.OUTPUT_DATA, "deepar_stability_study"))

MODEL_STABILITY_STUDY_PLOTS = files.create_folder(os.path.join(STABILITY_STUDY_PATH, "model_stability_study_plots"))


def model_stability_study_results_path(fixed_seeds=False):
    if fixed_seeds:
        stability_results_file = "fixed_seeds_model_stability_study_results.csv"
    else:
        stability_results_file = "model_stability_study_results.csv"

    return os.path.join(STABILITY_STUDY_PATH, stability_results_file)


NUM_EVAL_SAMPLES_STABILITY_STUDY_PLOTS = files.create_folder(
    os.path.join(STABILITY_STUDY_PATH, "num_eval_samples_stability_study_plots"))


def run_model_stability_study(max_epoch_list, nb_trials, fixed_seeds=False):
    """
    Run evaluation of same models trained nb_trials times with different max_epoch to assess deepar stability.

    :param max_epoch_list: list of max_epoch used in the various models trained.
    :param nb_trials: number of times that each model with given max_epoch has been trained.
    :return: Saves dataframe with results and plots of predictions for each model.
    """
    region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
    stability_study_results = []

    for max_epoch in max_epoch_list:
        logging.info(f"Running prediction with {nb_trials} models trained with {max_epoch} epochs.")
        for trial_nb in range(1, nb_trials + 1):
            forecasts, tss, model_pkl_path = prepare_data_for_deepar_plot(
                region_df_dict, [md.IDF], None, max_epoch, md.LEARNING_RATE, trial_nb, fixed_seeds=fixed_seeds)

            fig_path = os.path.join(MODEL_STABILITY_STUDY_PLOTS, f"{Path(model_pkl_path).name}.png")
            mape = plot_deepar_forecasts(
                region_df_dict, tss, forecasts, past_length=2 * md.NB_HOURS_PRED, fig_path=fig_path,
                prediction_date=md.END_TRAIN_DATE)
            result_dict = {"learning_rate": md.LEARNING_RATE,
                           "max_epoch": max_epoch,
                           "trial_nb": trial_nb,
                           "prediction_date": md.END_TRAIN_DATE,
                           "nb_days_pred": int(md.NB_HOURS_PRED / 24),
                           "MAPE": mape}

            stability_study_results.append(result_dict)

    stability_study_results_df = pd.DataFrame.from_records(stability_study_results)

    stability_study_results_df.to_csv(model_stability_study_results_path(fixed_seeds), index=False)


def run_num_eval_samples_stability_study(max_epoch, trial_nb, nb_pred, results_file,
                                         num_eval_samples_list, nb_additional_pred_days=0):
    region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
    stability_study_results = []

    model_pkl_path = predictor_path(region_df_dict, [md.IDF], max_epoch, md.LEARNING_RATE, None, trial_nb)

    with open(model_pkl_path, "rb") as model_pkl:
        deepar_model = pickle.load(model_pkl)

    for num_eval_samples in num_eval_samples_list:
        logging.info(
            f"Running prediction {nb_pred} time(s) on {nb_additional_pred_days + 1} day(s) with model trained with"
            f" {max_epoch} epochs and num_eval_samples = {num_eval_samples}.")
        for pred_nb in range(1, nb_pred + 1):
            for additional_day in range(nb_additional_pred_days + 1):
                prediction_date = md.END_TRAIN_DATE + timedelta(days=14*additional_day)
                forecasts, tss = make_predictions(
                    deepar_model, region_df_dict, prediction_date, [md.IDF], target_col=c.EnergyConso.CONSUMPTION,
                    feat_dynamic_cols=None, num_eval_samples=num_eval_samples)

                prediction_date_str = datetime.strftime(prediction_date, "%Y-%m-%d")
                fig_path = os.path.join(
                    NUM_EVAL_SAMPLES_STABILITY_STUDY_PLOTS,
                    f"{Path(model_pkl_path).name}_{prediction_date_str}_{num_eval_samples}_samples_{str(pred_nb)}.png")
                mape = plot_deepar_forecasts(
                    region_df_dict, tss, forecasts, past_length=2 * md.NB_HOURS_PRED, fig_path=fig_path,
                    prediction_date=prediction_date)
                result_dict = {"learning_rate": md.LEARNING_RATE,
                               "max_epoch": max_epoch,
                               "trial_nb": trial_nb,
                               "prediction_date": prediction_date,
                               "nb_days_pred": int(md.NB_HOURS_PRED / 24),
                               "num_eval_samples": num_eval_samples,
                               "pred_nb": pred_nb,
                               "MAPE": mape}

                stability_study_results.append(result_dict)

    stability_study_results_df = pd.DataFrame.from_records(stability_study_results)

    stability_study_results_df.to_csv(os.path.join(STABILITY_STUDY_PATH, results_file), index=False)


def plot_model_stability_study_results(max_epoch_list, nb_trials):
    results = pd.read_csv(model_stability_study_results_path(fixed_seeds=False))
    results = results[results["max_epoch"].isin(max_epoch_list) & (results["trial_nb"] <= nb_trials)]
    # Turn MAPE into percentage
    results["MAPE"] = results["MAPE"]*100

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(1, figsize=(20, 8))
    sns.boxplot(x="max_epoch", y="MAPE", data=results, whis=[10, 90])
    sns.swarmplot(x="max_epoch", y="MAPE", data=results, color=".25", s=10)
    plt.xlabel("Number of training epochs")
    plt.ylabel("MAPE (%)")
    # plt.ylim([0, 30])
    plt.title(f"Distribution of MAPE on January 1st prediction for {nb_trials} trainings")
    plt.savefig(os.path.join(STABILITY_STUDY_PATH, "model_stability_boxplot.png"))

    plt.close()


def free_vs_fixed_seeds_plot():
    results_free_seeds = pd.read_csv(model_stability_study_results_path(fixed_seeds=False))
    results_fixed_seeds = pd.read_csv(model_stability_study_results_path(fixed_seeds=True))

    results_free_seeds["seeds"] = "free"
    results_fixed_seeds["seeds"] = "fixed"
    results_free_seeds = pd.merge(results_free_seeds, results_fixed_seeds[["learning_rate", "max_epoch", "trial_nb"]],
                                  on=["learning_rate", "max_epoch", "trial_nb"], how="inner")

    all_results = pd.concat([results_free_seeds, results_fixed_seeds])

    # Turn MAPE into percentage
    all_results["MAPE"] = all_results["MAPE"] * 100

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(1, figsize=(20, 8))
    sns.boxplot(x="seeds", y="MAPE", data=all_results, whis=[10, 90])
    sns.swarmplot(x="seeds", y="MAPE", data=all_results, color=".25", s=10)
    plt.xlabel("Constraint on seeds")
    plt.ylabel("MAPE (%)")

    plt.title(f"Distribution of MAPE on January 1st prediction for models trained repeatedly with 20 iterations")
    plt.savefig(os.path.join(STABILITY_STUDY_PATH, "free_vs_fixed_seeds_model_stability_boxplot.png"))

    plt.close()

    
def plot_num_eval_samples_study_results(max_epoch, trial_nb, prediction_date):
    results = pd.read_csv(
        os.path.join(STABILITY_STUDY_PATH, files.NUM_EVAL_SAMPLES_STUDY_FILE), parse_dates=["prediction_date"])
    results = results[(results["max_epoch"] == max_epoch) & (results["trial_nb"] == trial_nb)
                      & (results["prediction_date"] == prediction_date)]

    # Turn MAPE into percentage
    results["MAPE_r"] = results["MAPE"].apply(lambda x: round(100*x, 2))

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(1, figsize=(20, 8))
    sns.boxplot(x="num_eval_samples", y="MAPE_r", data=results, whis=[10, 90])
    sns.swarmplot(x="num_eval_samples", y="MAPE_r", data=results, color=".25", s=10)
    plt.xlabel("Number of samples for prediction")
    plt.ylabel("MAPE (%)")
    plt.title("Distribution of MAPE on January 1st prediction for different eval sample sizes")
    plt.savefig(os.path.join(STABILITY_STUDY_PATH, "num_eval_samples_stability_boxplot.png"))

    plt.close()


def plot_model_performance_on_multiple_dates():
    results = pd.read_csv(os.path.join(STABILITY_STUDY_PATH, files.MULTIPLE_DAYS_PERFORMANCE_STUDY_FILE),
                         parse_dates=["prediction_date"])
    results["MAPE"] = results["MAPE"]*100

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(1, figsize=(15, 6))
    plt.scatter(results["prediction_date"], results["MAPE"])
    plt.ylabel("MAPE (%)")

    plt.title("MAPE for 2 week predictions from single model trained on 20 epochs")

    plt.savefig(os.path.join(STABILITY_STUDY_PATH, "model_performance_multiple_dates_scatterplot.png"))

    plt.close()


def plot_retraining_free_seeds_vs_multiple_dates():
    results_retraining = pd.read_csv(model_stability_study_results_path(fixed_seeds=False))
    results_multiple_dates = pd.read_csv(
        os.path.join(STABILITY_STUDY_PATH, files.MULTIPLE_DAYS_PERFORMANCE_STUDY_FILE),
        parse_dates=["prediction_date"])

    results_retraining["type"] = "models retrained with free seeds predicting a single date"
    results_multiple_dates["type"] = "single model predicting multiple dates"
    results_retraining = results_retraining[
        (results_retraining["max_epoch"] == results_multiple_dates["max_epoch"][0])
        & (results_retraining["learning_rate"] == results_multiple_dates["learning_rate"][0])]

    all_results = pd.concat([results_retraining, results_multiple_dates], sort=False)

    # Turn MAPE into percentage
    all_results["MAPE"] = all_results["MAPE"] * 100

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(1, figsize=(20, 8))
    sns.boxplot(x="type", y="MAPE", data=all_results, whis=[10, 90])
    sns.swarmplot(x="type", y="MAPE", data=all_results, color=".25", s=10)
    plt.xlabel("")
    plt.ylabel("MAPE (%)")

    plt.title(
        f"MAPE distributions for models retrained predicting single date and single model predicting multiple dates")
    plt.savefig(os.path.join(STABILITY_STUDY_PATH, "retraining_free_seeds_vs_multiple_dates_boxplot.png"))

    plt.close()

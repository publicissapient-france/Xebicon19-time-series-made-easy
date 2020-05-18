# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sys
sys.path.append("..")

# +
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

NUM_EVAL_SAMPLES_STABILITY_STUDY_RESULTS = os.path.join(
    STABILITY_STUDY_PATH, "num_eval_samples_stability_study_results.csv")


# -

def free_vs_fixed_seeds_plot():
    results_free_seeds = pd.read_csv(model_stability_study_results_path(fixed_seeds=False))
    results_fixed_seeds = pd.read_csv(model_stability_study_results_path(fixed_seeds=True))

    results_free_seeds["seeds"] = "free"
    results_fixed_seeds["seeds"] = "fixed"
    results_free_seeds = pd.merge(results_free_seeds, results_fixed_seeds[["learning_rate", "max_epoch", "trial_nb"]],
                           on=["learning_rate", "max_epoch", "trial_nb"], how="inner")

    all_results = pd.concat([results_free_seeds, results_fixed_seeds])

    # Turn MAPE into percentage
    all_results["MAPE"] = all_results["MAPE"]*100

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(1, figsize=(20, 8))
    ax = sns.boxplot(x="seeds", y="MAPE", data=all_results, whis=[10, 90])
    ax = sns.swarmplot(x="seeds", y="MAPE", data=all_results, color=".25", s=10)
    plt.xlabel("Constraint on seeds")
    plt.ylabel("MAPE (%)")
    
    plt.title(f"Distribution of MAPE on January 1st prediction for models trained with 20 iterations and fixed hyperparameters")
    #plt.savefig(os.path.join(STABILITY_STUDY_PATH, "free_vs_fixed_seeds_model_stability_boxplot.png"))

    plt.show()


free_vs_fixed_seeds_plot()


def plot_model_performance_on_multiple_dates()
    results = pd.read_csv(os.path.join(STABILITY_STUDY_PATH, files.MULTIPLE_DAYS_PERFORMANCE_STUDY_FILE),
                         parse_dates=["prediction_date"])
    results["MAPE"] = results["MAPE"]*100

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(1, figsize=(15, 6))
    plt.scatter(results["prediction_date"], results["MAPE"])
    plt.ylabel("MAPE (%)")

    plt.title("MAPE for 2 week predictions every 2 weeks with a single model trained on 20 epochs")

    plt.savefig(os.path.join(STABILITY_STUDY_PATH, "retraining_free_seeds_vs_multiple_dates_plot.png"))

    plt.show()


def retraining_free_seeds_vs_multiple_dates_plot():
    results_retraining = pd.read_csv(model_stability_study_results_path(fixed_seeds=False))
    results_multiple_dates = pd.read_csv(
        os.path.join(STABILITY_STUDY_PATH, files.MULTIPLE_DAYS_PERFORMANCE_STUDY_FILE),
        parse_dates=["prediction_date"])

    results_retraining["type"] = "models retrained with free seeds predicting a single date"
    results_multiple_dates["type"] = "single model predicting multiple dates"
    results_retraining = results_retraining[
        (results_retraining["max_epoch"]==results_multiple_dates["max_epoch"][0])
        &(results_retraining["learning_rate"]==results_multiple_dates["learning_rate"][0])]

    all_results = pd.concat([results_retraining, results_multiple_dates], sort=False)

    # Turn MAPE into percentage
    all_results["MAPE"] = all_results["MAPE"]*100

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(1, figsize=(20, 8))
    ax = sns.boxplot(x="type", y="MAPE", data=all_results, whis=[10, 90])
    ax = sns.swarmplot(x="type", y="MAPE", data=all_results, color=".25", s=10)
    plt.xlabel("")
    plt.ylabel("MAPE (%)")
    
    plt.title(f"MAPE distributions for models retrained predicting single date and single model predicting multiple dates")
    plt.savefig(os.path.join(STABILITY_STUDY_PATH, "retraining_free_seeds_vs_multiple_dates_plot.png"))

    plt.show()


retraining_free_seeds_vs_multiple_dates_plot()

# +
results_retraining = pd.read_csv(model_stability_study_results_path(fixed_seeds=False))
results_multiple_dates = pd.read_csv(
    os.path.join(STABILITY_STUDY_PATH, files.MULTIPLE_DAYS_PERFORMANCE_STUDY_FILE),
    parse_dates=["prediction_date"])

results_retraining["type"] = "retraining with free seeds to predict a single date"
results_multiple_dates["type"] = "single model prediction multiple dates"
results_retraining = pd.merge(results_retraining, results_multiple_dates[["learning_rate", "max_epoch"]],
                       on=["learning_rate", "max_epoch"], how="inner")

all_results = pd.concat([results_retraining, results_multiple_dates])
# -

results_retraining.head()

pd.set_option("display.max_rows", 100)
all_results.head(100)



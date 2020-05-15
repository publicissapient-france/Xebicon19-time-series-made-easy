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
    plt.savefig(os.path.join(STABILITY_STUDY_PATH, "free_vs_fixed_seeds_model_stability_boxplot.png"))

    plt.close()





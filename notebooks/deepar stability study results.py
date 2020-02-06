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

# +
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from time import time
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
pd.options.display.max_columns = None
pd.options.display.max_rows = None
matplotlib.rcParams.update({'font.size': 22})

import sys
sys.path.append("..")

import pickle

import src.constants.files as files
import src.constants.models as md
import src.constants.columns as c
from src.deepar.deepar_core import predictor_path, train_predictor, MODELS_PATH

from src.evaluation.evaluation import prepare_data_for_deepar_plot

import os
import re
# -

models = sorted(re.findall(r"Ile-de-France_[24681]0+_epochs_lr_0.0001_trial_\d0*", " ".join(os.listdir(MODELS_PATH))))

# +
region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
df_idf = region_df_dict[md.IDF]

for max_epochs in eval(md.DEEPAR_MAX_EPOCHS_LIST_STR):
    for trial_nb in range(1, 11):
        forecasts, tss, model_pkl_path = prepare_data_for_deepar_plot(
            region_df_dict, [md.IDF], None, max_epochs, md.LEARNING_RATE, trial_nb)
# -

results_path = "./deepar_stability_study/"

results = pd.read_csv(results_path + "results.csv", sep=",")

results.head()

# +
results["MAPE_pct"] = results["MAPE"].apply(lambda x: round(100*x, 2))

var_df = results.groupby("max_epochs").agg({"MAPE_pct": np.std})

var_df.head(100)

# +
import seaborn as sns

matplotlib.rcParams.update({'font.size': 22})
plt.figure(1, figsize=(20, 8))
ax = sns.boxplot(x="max_epochs", y="MAPE", data=results, whis=[10, 90])
ax = sns.swarmplot(x="max_epochs", y="MAPE", data=results, color=".25", s=10)
plt.xlabel("Number of training epochs")
plt.ylabel("MAPE (%)")
plt.ylim([0, 25])
plt.title("Distribution of MAPE on January 1st prediction for 10 trainings")
plt.savefig(results_path + "max_epochs_stability_boxplot.png")
# -

results_pred = pd.read_csv("./deepar_stability_study/sample_prediction_results.csv", sep=";")

results_pred.head()

results_pred["MAPE_r"] = results_pred["MAPE"].apply(lambda x: round(100*x, 2))
matplotlib.rcParams.update({'font.size': 22})
plt.figure(1, figsize=(20, 8))
ax = sns.boxplot(x="num_eval_sample", y="MAPE_r", data=results_pred, whis=[10, 90])
ax = sns.swarmplot(x="num_eval_sample", y="MAPE_r", data=results_pred, color=".25", s=10)
plt.xlabel("Number of samples for prediction")
plt.ylabel("MAPE (%)")
plt.ylim([5, 8])
plt.title("Distribution of MAPE on January 1st prediction for different sample sizes")
plt.savefig(results_path + "stability sample size.png")





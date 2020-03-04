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

# +
# %matplotlib inline
import matplotlib
import pickle
from fbprophet import Prophet
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from pytz import timezone as tz

from time import time
from datetime import date, datetime, timedelta

import sys
sys.path.append("..")

import src.constants.files as files

from src.prophet.prophet_core import *
import src.constants.files as files
import src.constants.models as md

from src.prophet.prophet_train import *
# -

with open(os.path.join(PROPHET_MODELS_PATH, files.PROPHET_2_YEARS_MODEL), "rb") as file:
    model_2_years = pickle.load(file)

future_idf_energy_date_2019 = model_2_years.make_future_dataframe(
    periods=md.NB_HOURS_PRED, freq=md.FREQ, include_history = False)
future_energy_forcast_idf_2019 = model_2_years.predict(future_idf_energy_date_2019)

# +
region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
df_idf = region_df_dict[md.IDF]

df_idf_plot = df_idf[(df_idf.index<=md.END_TRAIN_DATE + timedelta(hours=md.NB_HOURS_PRED))
                     &(df_idf.index>=md.START_TRAIN_DATE)].copy()

y_true = df_idf[(df_idf.index > md.END_TRAIN_DATE)
                &(df_idf.index <= md.END_TRAIN_DATE + timedelta(hours=md.NB_HOURS_PRED))]['Consommation (MW)']

y_pred = future_energy_forcast_idf_2019['yhat']


# -

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# +
matplotlib.rcParams.update({'font.size': 15})
plt.figure(figsize=(20,5),linewidth=2)

plt.plot(df_idf_plot[-2 * md.NB_HOURS_PRED:].index,
         df_idf_plot[-2 * md.NB_HOURS_PRED:]['Consommation (MW)'],
         color='steelblue',label='observation',linewidth=2.0)
plt.plot(future_energy_forcast_idf_2019.ds,
         future_energy_forcast_idf_2019.yhat,
         color='g',label='forecasting',linewidth=2.0)

plt.fill_between(future_energy_forcast_idf_2019.ds,
                 future_energy_forcast_idf_2019['yhat_lower'],
                 future_energy_forcast_idf_2019['yhat_upper'],
                 color='green', alpha=0.2, label='90% confidence interval' )

mape = mean_absolute_percentage_error(y_true, y_pred)
plt.title('Prophet: Prediction for Ile de France with' + " MAPE : {}%".format(str(round(100*mape, 1))))
plt.grid(which='both')
plt.ylim([12000, 28000])

plt.ylabel("consumption (MW)")
plt.rc('xtick', labelsize=10)  
plt.rc('ytick', labelsize=10)# fontsize of the axes title
plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=10)
plt.rc('legend', fontsize=10)
plt.margins(0)
plt.legend()
# -



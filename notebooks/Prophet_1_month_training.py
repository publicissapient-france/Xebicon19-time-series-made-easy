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

# +
start_train_date = datetime(2018, 1, 1)
end_train_date = datetime(2018, 12, 31, 23)

nb_days_pred = 3


# -

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# # Filter on Ile de France

region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))

df_idf = region_df_dict["Ile-de-France"]

df_idf.head()

# # Prophet for forcasting Ile de France energy

# +
df_idf_train = df_idf[(df_idf.index<=end_train_date)&(df_idf.index>=start_train_date)].copy()

df_idf_train["ds"] = df_idf_train.index
df_idf_train.rename(columns={"Consommation (MW)": "y"}, inplace=True)

df_prophet_train = df_idf_train[['ds','y']].reset_index(drop=True)
# -

start_time = time()
model_energy = Prophet(interval_width=0.9)
model_energy.fit(df_prophet_train)
end_time = time()
print ("This model took %.2f seconds to train" % (end_time - start_time))

future_idf_energy_date_2019 = model_energy.make_future_dataframe(
    periods=nb_days_pred*24,freq='H', include_history = False)

future_idf_energy_date_2019.head(2)

start_time = time()
future_energy_forcast_idf_2019 = model_energy.predict(future_idf_energy_date_2019)
end_time = time()
print ("This took %.2f seconds" % (end_time - start_time))

future_energy_forcast_idf_2019[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(2)

# +
df_idf_plot = df_idf[(df_idf.index<=end_train_date + timedelta(days=nb_days_pred))
                     &(df_idf.index>=start_train_date)].copy()

y_true = df_idf[(df_idf.index > end_train_date)
                &(df_idf.index <= end_train_date + timedelta(days=nb_days_pred))]['Consommation (MW)']

y_pred = future_energy_forcast_idf_2019['yhat']

# +
nb_days_plot = 7

matplotlib.rcParams.update({'font.size': 15})
plt.figure(figsize=(10,5),linewidth=2)

plt.plot(df_idf_plot[-nb_days_plot*24:].index,
         df_idf_plot[-nb_days_plot*24:]['Consommation (MW)'],
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


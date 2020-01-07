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
# Measure performance:

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# -

# # Filter on Ile de France

region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))

df_region_per_hour_ile_de_france = region_df_dict["Ile-de-France"]

plt.figure(figsize=(10,5))
plt.plot(df_region_per_hour_ile_de_france["date"], df_region_per_hour_ile_de_france["Consommation (MW)"])

# +
#restraindre la durée

# +
df_region_per_hour_ile_de_france_small = df_region_per_hour_ile_de_france[(df_region_per_hour_ile_de_france['date']>='2017-01-01 00:00:00')&(df_region_per_hour_ile_de_france['date']<'2019-01-15 00:00:00')]



# -

df_region_per_hour_ile_de_france_small.to_csv('../data/ile_de_france_dataset.csv')

# # Prophet for forcasting Ile de France energy

df_region_per_hour_ile_de_france_small.dtypes

df_region_per_hour_ile_de_france_small.head()

df_region_per_hour_ile_de_france_small['date_time']=pd.to_datetime(df_region_per_hour_ile_de_france_small['date'], format='%Y-%m-%d')

df_region_per_hour_ile_de_france_small.head()

df_region_per_hour_ile_de_france_small.dtypes

# +
# Select training set:
df_region_per_hour_ile_de_france_small_train = df_region_per_hour_ile_de_france_small[(df_region_per_hour_ile_de_france_small['date_time']<='2018-12-31 23:00:00')&(df_region_per_hour_ile_de_france_small['date_time']>='2018-12-01 00:00:00')]


# -

df_region_per_hour_ile_de_france_small_train.head()

df_region_per_hour_ile_de_france_small_train.tail()

model_energy = Prophet(interval_width=0.9)


df_region_per_hour_ile_de_france_small_train.rename(columns={"date_time": "ds"}, inplace=True)
df_region_per_hour_ile_de_france_small_train.rename(columns={"Consommation (MW)": "y"}, inplace=True)

df_prophet_train = df_region_per_hour_ile_de_france_small_train[['ds','y']]

df_prophet_train.head()

df_prophet_train.count()

# +
# It takes me 45 min to train a model  over 5 years !
# -

start_time = time()
model_energy.fit(df_prophet_train)
end_time = time()
print ("This model took %.2f seconds to train" % (end_time - start_time))

future_ile_de_france_energy_date_2019 = model_energy.make_future_dataframe(periods=3*24,freq='H', include_history = False)


future_ile_de_france_energy_date_2019.head(100)

start_time = time()
future_energy_forcast_ile_de_france_2019 = model_energy.predict(future_ile_de_france_energy_date_2019)
end_time = time()
print ("This took %.2f seconds" % (end_time - start_time))

future_energy_forcast_ile_de_france_2019[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].count()

# df_region_per_day_ile_de_france_2017_2018.head()

future_energy_forcast_ile_de_france_2019_plot= future_energy_forcast_ile_de_france_2019

df_region_per_hour_ile_de_france_plot=df_region_per_hour_ile_de_france_small

import matplotlib

# +
matplotlib.rcParams.update({'font.size': 15})
plt.figure(figsize=(10,5),linewidth=2)
plt.plot(df_region_per_hour_ile_de_france_plot[-600:-258].date,df_region_per_hour_ile_de_france_plot[-600:-258]['Consommation (MW)'],color='steelblue',label='observation',linewidth=2.0)
plt.plot(future_energy_forcast_ile_de_france_2019_plot.ds.dt.to_pydatetime(),future_energy_forcast_ile_de_france_2019_plot.yhat,color='g',label='forecasting',linewidth=2.0)

plt.fill_between(future_energy_forcast_ile_de_france_2019_plot.ds.dt.to_pydatetime(), future_energy_forcast_ile_de_france_2019_plot['yhat_lower'], future_energy_forcast_ile_de_france_2019_plot['yhat_upper'],
                        color='green', alpha=0.2,label='90% confidence interval' )
plt.title('Prophet: Prediction for Ile de France with' + " MAPE : {}%".format(str(round(100*0.17788634, 1))))
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



y_true = df_region_per_hour_ile_de_france_small[(df_region_per_hour_ile_de_france_small['date_time'] >= '2019-01-01')&(df_region_per_hour_ile_de_france_small['date_time'] < '2019-01-04')]['Consommation (MW)']


y_true.count()

y_pred = future_energy_forcast_ile_de_france_2019['yhat']

y_pred.count()

mean_absolute_percentage_error(y_true, y_pred)



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

import pandas as pd
from fbprophet import Prophet

# %matplotlib inline
from fbprophet import Prophet
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from pytz import timezone as tz

from time import time
from datetime import date
from datetime import datetime


# +
# Measure performance:

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# -

df_energy = pd.read_csv('../data/'+ "eco2mix-regional-cons-def.csv", sep=";",
                parse_dates=["Date - Heure"]).sort_values(by=["Région", "Date - Heure"])

df_energy = df_energy[["Région", "Date - Heure", "Consommation (MW)"]]

df_energy.fillna(df_energy.dropna()["Consommation (MW)"].mean(), inplace=True)

df_energy.isnull().sum()

df_energy.dtypes

df_energy.fillna(df_energy.dropna()["Consommation (MW)"].mean(), inplace=True)
df_energy["date"] = df_energy["Date - Heure"].apply(lambda x: x + timedelta(minutes=x.minute))
df_energy["date"] = df_energy["date"].apply(lambda x: x.astimezone(tz("UTC")))
df_energy["date"] = df_energy["date"].apply(lambda x: x.fromtimestamp(x.timestamp()))
df_energy.head(3)

df_energy.head()

# group by day:
df_region_per_hour = df_energy.groupby(["Région", "date"], as_index=False).agg({"Consommation (MW)": np.sum})

df_region_per_hour.head()

df_region_per_hour.dtypes

# # Filter on Ile de France

df_region_per_hour_ile_de_france = df_region_per_hour[df_region_per_hour['Région']=='Ile-de-France']

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

# Select training set:
df_region_per_hour_ile_de_france_small_train = df_region_per_hour_ile_de_france_small[(df_region_per_hour_ile_de_france_small['date_time']<='2018-12-31 23:00:00')&(df_region_per_hour_ile_de_france_small['date_time']>='2018-11-01 00:00:00')]

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

future_ile_de_france_energy_date_2019 = model_energy.make_future_dataframe(periods=14*24,freq='H', include_history = False)


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
plt.plot(df_region_per_hour_ile_de_france_plot[-650:].date,df_region_per_hour_ile_de_france_plot[-650:]['Consommation (MW)'],color='steelblue',label='observation',linewidth=2.0)
plt.plot(future_energy_forcast_ile_de_france_2019_plot.ds.dt.to_pydatetime(),future_energy_forcast_ile_de_france_2019_plot.yhat,color='g',label='forecasting',linewidth=2.0)

plt.fill_between(future_energy_forcast_ile_de_france_2019_plot.ds.dt.to_pydatetime(), future_energy_forcast_ile_de_france_2019_plot['yhat_lower'], future_energy_forcast_ile_de_france_2019_plot['yhat_upper'],
                        color='green', alpha=0.2,label='90% confidence interval' )
plt.title('Prophet: Prediction for Ile de France with' + " MAPE : {}%".format(str(round(100*0.058379, 1))))
plt.grid(which='both')
plt.ylim([12000, 28000])

plt.ylabel("consumption (MW)")
plt.rc('xtick', labelsize=9)  
plt.rc('ytick', labelsize=9)# fontsize of the axes title
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=9)
plt.rc('legend', fontsize=9)
plt.margins(0)
plt.legend()

# -



y_true = df_region_per_hour_ile_de_france_small[(df_region_per_hour_ile_de_france_small['date_time'] >= '2019-01-01')&(df_region_per_hour_ile_de_france_small['date_time'] < '2019-01-15')]['Consommation (MW)']


y_pred = future_energy_forcast_ile_de_france_2019['yhat']

mean_absolute_percentage_error(y_true, y_pred)



# # Add meteo:

model_energy_meteo = Prophet()

model_energy_meteo.add_regressor('max_temp_paris')

df_temp_train.rename(columns={"date": "ds"}, inplace=True)
df_temp_train.rename(columns={"Consommation (MW)": "y"}, inplace=True)

df_temp_train.head()

model_energy_meteo.fit(df_temp_train)

future_ile_de_france_energy_2019_meteo_date = model_energy_meteo.make_future_dataframe(periods=14*24,freq='H', include_history = False)


future_ile_de_france_energy_2019_meteo_date['max_temp_paris']=df_temp[(df_temp['date']>='2019-01-01')&(df_temp['date']<'2019-01-15')]['max_temp_paris'].to_numpy()

future_ile_de_france_energy_2019_meteo_date.tail()

future_energy_forcast_ile_de_france_2019_meteo = model_energy_meteo.predict(future_ile_de_france_energy_2019_meteo_date)

future_energy_forcast_ile_de_france_2019_meteo[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)

future_energy_forcast_ile_de_france_2019_meteo_plot= future_energy_forcast_ile_de_france_2019_meteo


future_energy_forcast_ile_de_france_2019_meteo_plot.tail()

import matplotlib

# +
matplotlib.rcParams.update({'font.size': 22})
plt.figure(figsize=(20,8),linewidth=2)
plt.plot(future_energy_forcast_ile_de_france_2019_meteo_plot.ds.dt.to_pydatetime(),future_energy_forcast_ile_de_france_2019_meteo_plot.yhat,color='green',label='forecasting',linewidth=2.0)
plt.plot(df_region_per_hour_ile_de_france_plot[-650:].date,df_region_per_hour_ile_de_france_plot[-650:]['Consommation (MW)'],color='steelblue',label='real data',linewidth=2.0)
plt.fill_between(future_energy_forcast_ile_de_france_2019_plot.ds.dt.to_pydatetime(), future_energy_forcast_ile_de_france_2019_plot['yhat_lower'], future_energy_forcast_ile_de_france_2019_plot['yhat_upper'],
                        color='green', alpha=0.2, label='90% confidence interval')
plt.title('Ile de France' + " MAPE : {}%".format(str(round(100*0.062957, 1))))
plt.grid(which='both')
plt.ylim([12000, 28000])

plt.ylabel("consumption (MW)")
plt.rc('xtick', labelsize=16)  
plt.rc('ytick', labelsize=16)# fontsize of the axes title
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=16)
plt.rc('legend', fontsize=16)
plt.margins(0)
plt.legend()
# -

matplotlib.rcParams.update({'font.size': 22})
plt.figure(figsize=(35,10),linewidth=2)
plt.plot(future_energy_forcast_ile_de_france_2019_meteo_plot.ds.dt.to_pydatetime(),future_energy_forcast_ile_de_france_2019_meteo_plot.yhat,color='green',label='forcasting',linewidth=4.0)
plt.plot(df_region_per_hour_ile_de_france_plot[-650:].date,df_region_per_hour_ile_de_france_plot[-650:]['Consommation (MW)'],color='dodgerblue',label='real data',linewidth=2.0)
plt.fill_between(future_energy_forcast_ile_de_france_2019_plot.ds.dt.to_pydatetime(), future_energy_forcast_ile_de_france_2019_plot['yhat_lower'], future_energy_forcast_ile_de_france_2019_plot['yhat_upper'],
                        color='#0072B2', alpha=0.2)
plt.title('Energy forcasting for Ile de France with temperature data 2019' + " - MAPE : {}%".format(str(round(100*0.062957, 1))))
plt.grid(which='both')
plt.legend()

# +
#measure performance:
# -

y_true = df_region_per_hour_ile_de_france_small[(df_region_per_hour_ile_de_france_small['date_time'] >= '2019-01-01')&(df_region_per_hour_ile_de_france_small['date_time'] < '2019-01-15')]['Consommation (MW)']


y_pred_meteo = future_energy_forcast_ile_de_france_2019_meteo['yhat']

mean_absolute_percentage_error(y_true, y_pred_meteo)





# +
# plot component ( how to interpret them ?)
# -

fig2 = model_energy.plot_components(future_energy_forcast_ile_de_france_2019)
fig2.set_size_inches(20,10,forward=True)

# ## Diagnostic: ( need to work on it again, I don't understand all)

# +
# Cross Validation 
# -

from < import cross_validation

cv = cross_validation(model_energy,initial='750 days', period='180 days', horizon = '356 days')

cv.tail()

from fbprophet.diagnostics import performance_metrics

df_pm = performance_metrics(cv)

df_pm.head()

from fbprophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(cv,metric='mape')



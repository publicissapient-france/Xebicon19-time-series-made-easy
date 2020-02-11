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
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

# %matplotlib inline
# -

energy = pd.read_csv('../data/ile_de_france_dataset.csv',
               index_col=['date'], parse_dates=['date']).sort_values(by=['date'])

energy.head()

energy= energy.rename(columns={"Consommation (MW)": "consommation"})

energy=energy.drop(columns=['Unnamed: 0','Région'])

energy.dtypes

energy.consommation = energy.consommation.astype(int)

energy.dtypes

energy_train = energy['2018-12-01 00:00:00':'2018-12-31 23:00:00']

energy_test =  energy['2019-01-01 00:00:00':'2019-01-03 23:00:00']

energy_train.tail()

energy_train.tail()

plt.figure(figsize=(15, 7))
plt.plot(energy_train.consommation)
plt.title('energy_train watched (hourly data)')
plt.grid(True)
plt.show()


# +
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# -

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


tsplot(energy_train.consommation, lags=60)


energy_diff = energy_train.consommation -energy_train.consommation.shift(24)
tsplot(energy_diff[24:], lags=60)

energy_diff = energy_diff - energy_diff.shift(1)
tsplot(energy_diff[24+1:], lags=60)

# +

ps = range(1, 5)
d=1 
qs = range(1, 5)
Ps = range(1, 3)
D=1 
Qs = range(1, 3)
s =24 # season length is still 24

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)


# -

def optimizeARIMA(parameters_list, d, D, s): 
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(energy_train.consommation, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table


# %%time
result_table = optimizeARIMA(parameters_list, d, D, s)


result_table.head()

# +
p, q, P, Q = result_table.parameters[0]

best_model=sm.tsa.statespace.SARIMAX(energy_train.consommation, order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())
# -

best_model.plot_diagnostics(figsize=(15, 12))

import matplotlib


def plotARIMA(serie_test,series, model, n_steps):
    """
        Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
        
    """
    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    print(data['arima_model'])
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['arima_model'][:s+d] = np.NaN
    
    pred_uc = model.get_forecast(steps=72)

    pred_ci = pred_uc.conf_int(alpha=0.1)
    print(pred_ci)
  
    
    # forecasting on n_steps forward 
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
    forecast = data.arima_model.append(forecast)
    forcast_compare= forecast['2019-01-01 00:00:00':'2019-01-03 23:00:00']
    print(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = np.mean(np.abs((serie_test.consommation - forcast_compare) / serie_test.consommation)) * 100
    matplotlib.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10, 5),linewidth=2)
    plt.title("SARIMA: Prediction for Ile de France with MAPE: {0:.2f}%".format(error))
    plt.plot(data.actual[-330:], color='steelblue',label='observation',linewidth=2.0)
    plt.plot(forcast_compare, color='g', label="forecasting",linewidth=2.0)
    #plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.3, color='lightgrey')
    plt.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='green', alpha=.2,label='90% confidence interval' )
 
    
    plt.plot(serie_test.consommation, color='steelblue',label='_nolegend_')
    print('serie_test.consommation',len(serie_test.consommation))
    print('forcast_compare',len(forcast_compare))
    plt.ylim([12000, 28000])
    plt.ylabel("consumption (MW)")
    plt.legend(loc=2)
    plt.margins(0)
    plt.rc('xtick', labelsize=10)  
    plt.rc('ytick', labelsize=10)# fontsize of the axes title
    plt.rc('axes', titlesize=15)
    plt.rc('axes', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.grid(True);

plotARIMA(energy_test,energy_train, best_model, 72)



import os
import pickle
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from src.sarima.sarima_train import MODELS_PATH as SARIMA_MODELS_PATH

import src.constants.files as files
import src.constants.models as md
import src.constants.columns as c


region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
idf_df = region_df_dict[md.IDF]
idf_df[c.EnergyConso.CONSUMPTION] = idf_df[c.EnergyConso.CONSUMPTION].fillna(idf_df[c.EnergyConso.CONSUMPTION].mean())

idf_train = idf_df[md.END_TRAIN_DATE - timedelta(days=365):md.END_TRAIN_DATE]
idf_test = idf_df[md.END_TRAIN_DATE:md.END_TRAIN_DATE + timedelta(hours=md.NB_HOURS_PRED)]

with open(os.path.join(SARIMA_MODELS_PATH, "best_model.pkl"), "rb") as file:
    best_model = pickle.load(file)


def plot_arima(serie_test, series, model, n_steps, s, d):
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
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['arima_model'][:s + d] = np.NaN

    pred_uc = model.get_forecast(steps=72)

    pred_ci = pred_uc.conf_int(alpha=0.1)

    # forecasting on n_steps forward
    forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data.arima_model.append(forecast)
    forcast_compare = forecast['2019-01-01 00:00:00':'2019-01-03 23:00:00']
    # calculate error, again having shifted on s+d steps from the beginning
    error = np.mean(np.abs((serie_test.consommation - forcast_compare) / serie_test.consommation)) * 100
    matplotlib.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10, 5), linewidth=2)
    plt.title("SARIMA: Prediction for Ile de France with MAPE: {0:.2f}%".format(error))
    plt.plot(data.actual[-330:], color='steelblue', label='observation', linewidth=2.0)
    plt.plot(forcast_compare, color='g', label="forecasting", linewidth=2.0)
    # plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.3, color='lightgrey')
    plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='green', alpha=.2,
                     label='90% confidence interval')

    plt.plot(serie_test.consommation, color='steelblue', label='_nolegend_')
    print('serie_test.consommation', len(serie_test.consommation))
    print('forcast_compare', len(forcast_compare))
    plt.ylim([12000, 28000])
    plt.ylabel("consumption (MW)")
    plt.legend(loc=2)
    plt.margins(0)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)  # fontsize of the axes title
    plt.rc('axes', titlesize=15)
    plt.rc('axes', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.grid(True)

plot_arima(idf_test,idf_train, best_model, md.NB_HOURS_PRED, s=1, d=1)

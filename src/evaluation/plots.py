import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import os
import numpy as np

import src.constants.columns as c
import src.constants.files as files
# TODO: Mettre toutes les fonctions de plot dans ce module


def plot_consumptions(region_df_dict, year, month):
    matplotlib.rcParams.update({'font.size': 22})

    plt.figure(1, figsize=(25, 12))
    for region in region_df_dict.keys():
        df_region = region_df_dict[region]
        df_region[c.EnergyConso.DATE_HEURE] = df_region.index
        df_region = df_region[(df_region[c.EnergyConso.DATE_HEURE].apply(lambda x: x.year)==year)
                           &(df_region[c.EnergyConso.DATE_HEURE].apply(lambda x: x.month==month))]
        plt.plot(df_region[c.EnergyConso.DATE_HEURE], df_region[c.EnergyConso.CONSUMPTION], label=region)
        plt.ylabel(c.EnergyConso.CONSUMPTION)

    plt.legend()
    plt.savefig(os.path.join(files.PLOTS, "Consos énergie France {}-{}.png".format(year, month)))

    plt.close()


def plot_arima(serie_test, series, model, n_steps):
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
    data['arima_model'][:s + d] = np.NaN

    pred_uc = model.get_forecast(steps=72)

    pred_ci = pred_uc.conf_int(alpha=0.1)
    print(pred_ci)

    # forecasting on n_steps forward
    forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data.arima_model.append(forecast)
    forcast_compare = forecast['2019-01-01 00:00:00':'2019-01-03 23:00:00']
    print(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = np.mean(np.abs((serie_test[c.EnergyConso.CONSUMPTION] - forcast_compare) / serie_test[c.EnergyConso.CONSUMPTION])) * 100
    matplotlib.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10, 5), linewidth=2)
    plt.title("SARIMA: Prediction for Ile de France with MAPE: {0:.2f}%".format(error))
    plt.plot(data.actual[-330:], color='steelblue', label='observation', linewidth=2.0)
    plt.plot(forcast_compare, color='g', label="forecasting", linewidth=2.0)
    # plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.3, color='lightgrey')
    plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='green', alpha=.2,
                     label='90% confidence interval')

    plt.plot(serie_test[c.EnergyConso.CONSUMPTION], color='steelblue', label='_nolegend_')
    print('serie_test[c.EnergyConso.CONSUMPTION]', len(serie_test[c.EnergyConso.CONSUMPTION]))
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
    plt.grid(True);
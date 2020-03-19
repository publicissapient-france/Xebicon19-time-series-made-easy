import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

import statsmodels.tsa.api as smt
import statsmodels.api as sm

import src.constants.columns as c


def tsplot(y, lags=None, figsize=(12, 7), style='bmh', filename=None):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    pd.plotting.register_matplotlib_converters()

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
        if filename is not None:
            plt.savefig(filename)
        plt.close()


def optimize_arima(idf_train, parameters_list, d, D, s):
    results = []
    best_aic = float("inf")

    for param in parameters_list:
        # we need try-except because on some combinations model fails to converge
        try:
            logging.info(
                f"Fitting SARIMAX with order {(param[0], d, param[1])} and seasonal order {(param[2], D, param[3], s)}")
            model = sm.tsa.statespace.SARIMAX(idf_train[c.EnergyConso.CONSUMPTION], order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            logging.info("Model failed to converge.")
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

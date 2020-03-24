from datetime import datetime, timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from src.core import mean_absolute_percentage_error
import src.constants.columns as c
import src.constants.files as files
import src.constants.models as md
from src.sarima.sarima_train import SARIMA_MODELS_PATH as SARIMA_MODELS_PATH


DEEPAR_PLOTS = files.create_folder(os.path.join(files.PLOTS, "deepar"))
PROPHET_PLOTS = files.create_folder(os.path.join(files.PLOTS, "prophet"))
SARIMA_PLOTS = files.create_folder(os.path.join(files.PLOTS, "sarima"))

LABEL_FONTSIZE = 16
TITLE_FONTSIZE = 20


def plot_consumptions(region_df_dict, year, month):
    matplotlib.rcParams.update({'font.size': 22})

    plt.figure(1, figsize=(25, 12))
    for region in region_df_dict.keys():
        df_region = region_df_dict[region]
        df_region[c.EnergyConso.DATE_HEURE] = df_region.index
        df_region = df_region[(df_region[c.EnergyConso.DATE_HEURE].apply(lambda x: x.year)==year)
                           &(df_region[c.EnergyConso.DATE_HEURE].apply(lambda x: x.month==month))]
        plt.plot(df_region[c.EnergyConso.DATE_HEURE], df_region[c.EnergyConso.CONSUMPTION], label=region)
        plt.ylabel("Consumption (MW)")

    plt.legend()
    plt.savefig(os.path.join(files.PLOTS, "Consos énergie France {}-{}.png".format(year, month)))

    plt.close()


def plot_deepar_forecasts(df_dict, tss, forecasts, past_length, fig_path):
    register_matplotlib_converters()

    target = tss[0]
    forecast = forecasts[0]
    ax = target[-past_length:].plot(figsize=(20, 5), linewidth=2)
    forecast.plot(color='g')
    plt.grid(which='both')
    plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])

    results_mean = forecast.mean
    ground_truth = df_dict[forecast.item_id][c.EnergyConso.CONSUMPTION][
                   md.END_TRAIN_DATE + timedelta(hours=1):md.END_TRAIN_DATE + timedelta(hours=md.NB_HOURS_PRED)].values
    mape = mean_absolute_percentage_error(ground_truth, results_mean)

    plt.title("Deepar: Prediction for " + forecast.item_id + " with MAPE: {}%".format(str(round(100 * mape, 1))),
              fontsize=TITLE_FONTSIZE)
    plt.ylabel("Consumption (MW)", fontsize=LABEL_FONTSIZE)
    plt.xlabel("")
    ax.set_xlim([md.END_TRAIN_DATE - timedelta(days=md.NB_HOURS_PRED / 24),
                 md.END_TRAIN_DATE + timedelta(days=md.NB_HOURS_PRED / 24)])
    ax.set_ylim([12000, 28000])
    xticks = [md.END_TRAIN_DATE + timedelta(days=x) for x in [-11, -7, -3, 0, 4, 8, 12]]
    ax.set_xticks(xticks, minor=True)
    ax.set_xticklabels([datetime.strftime(date, "%Y-%m-%d") for date in xticks if date != md.END_TRAIN_DATE],
                       minor=True, fontsize=LABEL_FONTSIZE)
    ax.set_xticklabels(["", datetime.strftime(md.END_TRAIN_DATE, "%Y-%m-%d"), ""], minor=False,
                       fontsize=LABEL_FONTSIZE)
    yticks = np.arange(14000, 28000, step=2000)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(x) for x in yticks], fontsize=LABEL_FONTSIZE)
    plt.savefig(fig_path)
    plt.close()

    return mape


def plot_prophet_forecast(energy_forecast_idf, df_idf_plot, mape, figname):
    plt.figure(figsize=(20, 5), linewidth=2)

    forecast = energy_forecast_idf[energy_forecast_idf["ds"] > md.END_TRAIN_DATE]
    model_past_fitting = energy_forecast_idf[energy_forecast_idf["ds"] <= md.END_TRAIN_DATE]

    plt.plot(
        df_idf_plot.index, df_idf_plot[c.EnergyConso.CONSUMPTION], color="steelblue", label="observations", linewidth=2)
    plt.plot(
        forecast["ds"], forecast["yhat"], color="g", label="forecasting", linewidth=2)

    if len(model_past_fitting) > 0:
        plt.plot(model_past_fitting["ds"], model_past_fitting["yhat"], color="g", label="fitting of past values",
                 linestyle="dashed", linewidth=2)

    plt.fill_between(forecast["ds"],
                     forecast["yhat_lower"],
                     forecast["yhat_upper"],
                     color="green", alpha=0.2, label="90% confidence interval" )
    plt.title("Prophet: Prediction for Ile de France with" + " MAPE: {}%".format(str(round(100 * mape, 1))),
              fontsize=TITLE_FONTSIZE)
    plt.grid(which="both")
    plt.ylabel("Consumption (MW)", fontsize=LABEL_FONTSIZE)

    ax = plt.gca()
    ax.set_xlim([md.END_TRAIN_DATE - timedelta(days=md.NB_HOURS_PRED / 24),
                 md.END_TRAIN_DATE + timedelta(days=md.NB_HOURS_PRED / 24)])
    ax.set_ylim([12000, 28000])
    xticks = [md.END_TRAIN_DATE + timedelta(days=x) for x in [-11, -7, -3, 0, 4, 8, 12]]
    ax.set_xticks(xticks, minor=False)
    ax.set_xticklabels([datetime.strftime(date, "%Y-%m-%d") for date in xticks], minor=False, fontsize=LABEL_FONTSIZE)
    yticks = np.arange(14000, 28000, step=2000)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(x) for x in yticks], fontsize=LABEL_FONTSIZE)

    plt.legend()

    plt.savefig(os.path.join(PROPHET_PLOTS, "{}.png".format(figname.replace(".pkl", ""))))

    plt.close()


def plot_sarima_forecast(idf_df):
    idf_df[c.EnergyConso.CONSUMPTION] = idf_df[c.EnergyConso.CONSUMPTION].fillna(
        idf_df[c.EnergyConso.CONSUMPTION].mean())

    idf_train = idf_df[md.END_TRAIN_DATE - timedelta(days=365):md.END_TRAIN_DATE]
    idf_test = idf_df[md.END_TRAIN_DATE:md.END_TRAIN_DATE + timedelta(hours=md.NB_HOURS_PRED)]

    best_model = SARIMAXResults.load(os.path.join(SARIMA_MODELS_PATH, "best_model.pkl"))

    plot_sarima(idf_test, idf_train, best_model, md.NB_HOURS_PRED, s=1, d=1)


def plot_sarima(serie_test, series, model, n_steps, s, d):
    """
        Plots model vs predicted values

        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future

    """
    # adding model values
    data = series[[c.EnergyConso.CONSUMPTION]].copy()
    data['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['arima_model'][:s + d] = np.NaN

    pred_uc = model.get_forecast(steps=n_steps)

    pred_ci = pred_uc.conf_int(alpha=0.1)

    # forecasting on n_steps forward
    forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data.arima_model.append(forecast)
    forcast_compare = forecast[md.END_TRAIN_DATE:md.END_TRAIN_DATE + timedelta(hours=n_steps)]
    # calculate error, again having shifted on s+d steps from the beginning
    error = np.mean(np.abs((serie_test[c.EnergyConso.CONSUMPTION] - forcast_compare)
                           / serie_test[c.EnergyConso.CONSUMPTION])) * 100
    matplotlib.rcParams.update({'font.size': 15})
    plt.figure(figsize=(20, 5), linewidth=2)
    plt.title("SARIMA: Prediction for Ile de France with MAPE: {0:.1f}%".format(error))
    plt.plot(data[c.EnergyConso.CONSUMPTION][-330:], color='steelblue', label='observation', linewidth=2.0)
    plt.plot(forcast_compare, color='g', label="forecasting", linewidth=2.0)
    # plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.3, color='lightgrey')
    plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='green', alpha=.2,
                     label='90% confidence interval')

    plt.plot(serie_test[c.EnergyConso.CONSUMPTION], color='steelblue', label='_nolegend_')
    plt.ylim([12000, 28000])
    plt.ylabel("Consumption (MW)")
    plt.legend(loc=2)
    plt.margins(0)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)  # fontsize of the axes title
    plt.rc('axes', titlesize=15)
    plt.rc('axes', labelsize=10)
    plt.rc('legend', fontsize=10)
    plt.grid(True)

    plt.savefig(os.path.join(SARIMA_PLOTS, f"sarima_pred_{int(n_steps/24)}_days.png"))
    plt.close()

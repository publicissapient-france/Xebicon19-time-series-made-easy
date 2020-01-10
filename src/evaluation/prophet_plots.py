import matplotlib
import matplotlib.pyplot as plt
import os

import src.constants.models as md
import src.constants.files as files
import src.constants.columns as c

PROPHET_PLOTS = files.create_folder(os.path.join(files.PLOTS, "prophet"))


def plot_prophet_forecast(energy_forecast_idf, df_idf_plot, mape, figname):
    matplotlib.rcParams.update({"font.size": 15})
    plt.figure(figsize=(20 ,5) ,linewidth=2)

    plt.plot(df_idf_plot.index,
             df_idf_plot[c.EnergyConso.CONSUMPTION],
             color="steelblue" ,label="observation" ,linewidth=2.0)
    plt.plot(energy_forecast_idf["ds"],
             energy_forecast_idf["yhat"],
             color="g" ,label="forecasting" ,linewidth=2.0)

    plt.fill_between(energy_forecast_idf["ds"],
                     energy_forecast_idf["yhat_lower"],
                     energy_forecast_idf["yhat_upper"],
                     color="green", alpha=0.2, label="90% confidence interval" )
    plt.title("Prophet: Prediction for Ile de France with" + " MAPE : {}%".format(str(round(100 * mape, 1))))
    plt.grid(which="both")
    plt.ylim([12000, 28000])

    plt.ylabel("consumption (MW)")
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("axes", titlesize=15)
    plt.rc("axes", labelsize=10)
    plt.rc("legend", fontsize=10)
    plt.margins(0)
    plt.legend()

    plt.savefig(os.path.join(PROPHET_PLOTS, "{}.png".format(figname.replace(".pkl", ""))))

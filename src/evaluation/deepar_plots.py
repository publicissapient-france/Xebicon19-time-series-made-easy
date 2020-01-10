import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import os

import src.constants.models as md
import src.constants.files as files
import src.constants.columns as c
from src.core import mean_absolute_percentage_error


DEEPAR_PLOTS = files.create_folder(os.path.join(files.PLOTS, "deepar"))


def plot_forecasts(df_dict, test_date, tss, forecasts, past_length, figname):
    label_fontsize = 16
    for target, forecast in zip(tss, forecasts):
        ax = target[-past_length:].plot(figsize=(20, 8), linewidth=2)
        forecast.plot(color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])

        results_mean = forecast.mean
        ground_truth = df_dict[forecast.item_id][c.EnergyConso.CONSUMPTION][
                       test_date + timedelta(hours=1):test_date + timedelta(hours=md.NB_HOURS_PRED)].values
        mape = mean_absolute_percentage_error(ground_truth, results_mean)

        plt.title(forecast.item_id + " MAPE:{}%".format(str(round(100 * mape, 1))))
        plt.ylabel(c.EnergyConso.CONSUMPTION)
        plt.xlabel("")
        ax.set_xlim([test_date - timedelta(days=md.NB_HOURS_PRED / 24), test_date + timedelta(days=md.NB_HOURS_PRED / 24)])
        ax.set_ylim([12000, 28000])
        xticks = [test_date + timedelta(days=x) for x in [-11, -7, -3, 0, 4, 8, 12]]
        ax.set_xticks(xticks, minor=True)
        ax.set_xticklabels([datetime.strftime(date, "%Y-%m-%d") for date in xticks if date != test_date],
                           minor=True, fontsize=label_fontsize)
        ax.set_xticklabels(["", datetime.strftime(test_date, "%Y-%m-%d"), ""], minor=False,
                           fontsize=label_fontsize)
        yticks = np.arange(14000, 28000, step=2000)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(x) for x in yticks], fontsize=label_fontsize)
        plt.savefig(os.path.join(DEEPAR_PLOTS, f"{figname}.png"))
        plt.close()

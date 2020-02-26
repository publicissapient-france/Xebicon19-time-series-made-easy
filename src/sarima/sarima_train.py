from datetime import timedelta
from itertools import product
import pickle
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt

import src.constants.files as files
import src.constants.models as md
import src.constants.columns as c

from src.sarima.sarima_core import tsplot, optimize_arima

SARIMA_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = files.create_folder(os.path.join(SARIMA_PATH, "models" + files.TEST_SUFFIX))
PLOTS_PATH = files.create_folder(os.path.join(SARIMA_PATH, "plots" + files.TEST_SUFFIX))


def sarima_train():
    region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
    idf_df = region_df_dict[md.IDF]
    idf_df[c.EnergyConso.CONSUMPTION] = idf_df[c.EnergyConso.CONSUMPTION].fillna(idf_df[c.EnergyConso.CONSUMPTION].mean())

    idf_train = idf_df[md.END_TRAIN_DATE - timedelta(days=365):md.END_TRAIN_DATE]

    plot_stat_tests(idf_train)

    ps = range(1, int(os.getenv("MAX_ARIMA_PARAM_RANGE", 4)))
    d = 1
    qs = range(1, int(os.getenv("MAX_ARIMA_PARAM_RANGE", 3)))
    Ps = range(1, int(os.getenv("MAX_ARIMA_PARAM_RANGE", 3)))
    D = 1
    Qs = range(1, int(os.getenv("MAX_ARIMA_PARAM_RANGE", 3)))
    s = 24  # season length is still 24

    # creating list with all the possible combinations of parameters
    parameters_list = list(product(ps, qs, Ps, Qs))

    result_table = optimize_arima(idf_train, parameters_list, d, D, s)

    result_table.to_csv(os.path.join(MODELS_PATH, "arima_optimization_results.csv"), index=False)

    p, q, P, Q = result_table.parameters[0]

    best_model = sm.tsa.statespace.SARIMAX(idf_train[c.EnergyConso.CONSUMPTION], order=(p, d, q),
                                           seasonal_order=(P, D, Q, s)).fit(disp=-1)

    with open(os.path.join(MODELS_PATH, "best_model_summary.txt"), "w") as file:
        file.write(best_model.summary().as_csv())

    with open(os.path.join(MODELS_PATH, "best_model.pkl"), "wb") as file:
        pickle.dump(best_model, file)

    figure = plt.figure(1, figsize=(15, 12))
    best_model.plot_diagnostics(fig=figure)
    plt.savefig(os.path.join(PLOTS_PATH, "best_model_diagnostic.png"))
    plt.close()


def plot_stat_tests(idf_train):
    tsplot(idf_train[c.EnergyConso.CONSUMPTION], lags=60, filename=os.path.join(PLOTS_PATH, "tsplot_train_data.png"))

    idf_df_diff = idf_train[c.EnergyConso.CONSUMPTION] - idf_train[c.EnergyConso.CONSUMPTION].shift(24)
    tsplot(idf_df_diff[24:], lags=60, filename=os.path.join(PLOTS_PATH, "tsplot_diff_24.png"))

    idf_df_diff = idf_df_diff - idf_df_diff.shift(1)
    tsplot(idf_df_diff[24 + 1:], lags=60, filename=os.path.join(PLOTS_PATH, "tsplot_diff_24.png"))

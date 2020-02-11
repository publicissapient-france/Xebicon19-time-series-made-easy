from datetime import timedelta
from itertools import product
import pickle
import statsmodels.api as sm

import src.constants.files as files
import src.constants.models as md
import src.constants.columns as c

from src.sarima.sarima_core import tsplot, optimize_arima


region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
idf_df = region_df_dict[md.IDF]
idf_df[c.EnergyConso.CONSUMPTION] = idf_df[c.EnergyConso.CONSUMPTION].fillna(idf_df[c.EnergyConso.CONSUMPTION].mean())

idf_train = idf_df[md.END_TRAIN_DATE - timedelta(days=365):md.END_TRAIN_DATE]

idf_test = idf_df[md.END_TRAIN_DATE:md.END_TRAIN_DATE + timedelta(days=4)]

tsplot(idf_train[c.EnergyConso.CONSUMPTION], lags=60)

idf_df_diff = idf_train[c.EnergyConso.CONSUMPTION] - idf_train[c.EnergyConso.CONSUMPTION].shift(24)
tsplot(idf_df_diff[24:], lags=60)

idf_df_diff = idf_df_diff - idf_df_diff.shift(1)
tsplot(idf_df_diff[24 + 1:], lags=60)

ps = range(1, 5)
d = 1
qs = range(1, 5)
Ps = range(1, 3)
D = 1
Qs = range(1, 3)
s = 24  # season length is still 24

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

result_table = optimize_arima(parameters_list, d, D, s)

result_table.head()

p, q, P, Q = result_table.parameters[0]

best_model = sm.tsa.statespace.SARIMAX(idf_train[c.EnergyConso.CONSUMPTION], order=(p, d, q),
                                       seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())

best_model.plot_diagnostics(figsize=(15, 12))

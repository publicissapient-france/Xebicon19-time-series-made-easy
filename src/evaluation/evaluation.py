import pickle
import os
import logging
import pandas as pd

from datetime import timedelta
from pathlib import Path

import src.constants.files as files
import src.constants.models as md
import src.constants.columns as c
from src.core import mean_absolute_percentage_error
from src.prophet.prophet_train import PROPHET_MODELS_PATH
from src.evaluation.plots import plot_prophet_forecast
from src.evaluation.plots import plot_deepar_forecasts
from src.deepar.deepar_train import deepar_training_confs
from src.deepar.deepar_core import predictor_path, make_predictions
from src.sarima.sarima_train import SARIMA_MODELS_PATH as SARIMA_MODELS_PATH
from src.evaluation.plots import plot_sarima_forecast
from src.evaluation.plots import DEEPAR_PLOTS


def evaluate_models(deepar_max_epochs):
    region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
    df_idf = region_df_dict[md.IDF]

    logging.info("Plotting Prophet forecasts.")
    for model_name in [files.PROPHET_2_YEARS_MODEL, files.PROPHET_2_YEARS_WEATHER_MODEL]:
        df_idf_plot, mape, energy_forecast_idf = prepare_data_for_prophet_plot(df_idf, model_name)

        # Plot with forecasting only to show discontinuity of predictions.
        plot_prophet_forecast(
            energy_forecast_idf[energy_forecast_idf["ds"] > md.END_TRAIN_DATE], df_idf_plot, mape, figname=model_name)
        # Plot full model fitting to show hidden continuity.
        plot_prophet_forecast(energy_forecast_idf, df_idf_plot, mape, figname=model_name + "_full_fitting")

    logging.info("Plotting Deepar forecasts.")
    deepar_confs = deepar_training_confs(region_df_dict)
    for deepar_conf in deepar_confs:
        forecasts, tss, model_pkl_path = prepare_data_for_deepar_plot(
            region_df_dict, deepar_conf["region_list"], deepar_conf["feat_dynamic_cols"],
            deepar_max_epochs, md.LEARNING_RATE)

        fig_path = os.path.join(DEEPAR_PLOTS, f"{Path(model_pkl_path).name}.png")
        plot_deepar_forecasts(region_df_dict, tss, forecasts, past_length=2 * md.NB_HOURS_PRED, fig_path=fig_path)

    if os.path.exists(os.path.join(SARIMA_MODELS_PATH, "best_model.pkl")):
        logging.info("Plotting Sarima forecasts.")
        plot_sarima_forecast(df_idf)


def prepare_data_for_prophet_plot(df_idf, model_name):
    with open(os.path.join(PROPHET_MODELS_PATH, model_name), "rb") as file:
        model_energy = pickle.load(file)

    future_dates_for_forecast = model_energy.make_future_dataframe(
        periods=md.NB_HOURS_PRED, freq=md.FREQ, include_history=True)
    # Add temp covariate
    future_dates_for_forecast = pd.merge(
        future_dates_for_forecast, df_idf[[c.Meteo.MAX_TEMP_PARIS]], left_on="ds", right_index=True, how="left")

    energy_forecast_idf = model_energy.predict(future_dates_for_forecast)
    energy_forecast_idf = energy_forecast_idf[
        (energy_forecast_idf["ds"] <= md.END_TRAIN_DATE + timedelta(hours=md.NB_HOURS_PRED))
        & (energy_forecast_idf["ds"] >= md.END_TRAIN_DATE - timedelta(hours=md.NB_HOURS_PRED))].copy()

    df_idf_plot = df_idf[(df_idf.index <= md.END_TRAIN_DATE + timedelta(hours=md.NB_HOURS_PRED))
                         & (df_idf.index >= md.END_TRAIN_DATE - timedelta(hours=md.NB_HOURS_PRED))].copy()

    y_true = df_idf[
        (df_idf.index > md.END_TRAIN_DATE)
        & (df_idf.index < md.END_TRAIN_DATE + timedelta(hours=md.NB_HOURS_PRED))][c.EnergyConso.CONSUMPTION]
    y_pred = energy_forecast_idf[
        (energy_forecast_idf["ds"] > md.END_TRAIN_DATE)
        & (energy_forecast_idf["ds"] < md.END_TRAIN_DATE + timedelta(hours=md.NB_HOURS_PRED))]['yhat']

    mape = mean_absolute_percentage_error(y_true, y_pred)

    return df_idf_plot, mape, energy_forecast_idf


def prepare_data_for_deepar_plot(region_df_dict, regions_list, feat_dynamic_cols, max_epochs, learning_rate,
                                 trial_number=1, num_eval_samples=100):
    model_pkl_path = predictor_path(
        region_df_dict, regions_list, max_epochs, learning_rate, feat_dynamic_cols,
        trial_number)

    with open(model_pkl_path, "rb") as model_pkl:
        deepar_model = pickle.load(model_pkl)

    forecasts, tss = make_predictions(
        deepar_model, region_df_dict, md.END_TRAIN_DATE, [md.IDF], target_col=c.EnergyConso.CONSUMPTION,
        feat_dynamic_cols=feat_dynamic_cols, num_eval_samples=num_eval_samples)

    return forecasts, tss, model_pkl_path

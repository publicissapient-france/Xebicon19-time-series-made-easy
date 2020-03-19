import pickle
from time import time
import logging
import os

from src.prophet.prophet_core import format_training_data
import src.constants.files as files
import src.constants.models as md
import src.constants.columns as c

from fbprophet import Prophet

PROPHET_MODELS_PATH = files.create_folder(os.path.join(files.MODELS, "prophet"))


def prophet_train():
    logging.info("Preparing data for Prophet training.")
    region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
    df_dict = region_df_dict[md.IDF]

    df_prophet_train = format_training_data(df_dict, md.START_TRAIN_DATE, md.END_TRAIN_DATE)

    logging.info("Training Prophet model on 2 years.")
    start_time = time()
    model_energy = Prophet(yearly_seasonality=True)
    model_energy.fit(df_prophet_train)
    with open(os.path.join(PROPHET_MODELS_PATH, files.PROPHET_2_YEARS_MODEL), "wb") as file:
        pickle.dump(model_energy, file)
    logging.info("Training Prophet model on 2 years took %.2f seconds." % (time() - start_time))

    logging.info("Training Prophet model on 2 years with weather covariate.")
    start_time = time()
    model_energy_with_weather = Prophet(yearly_seasonality=True)
    model_energy_with_weather.add_regressor(c.Meteo.MAX_TEMP_PARIS)
    model_energy_with_weather.fit(df_prophet_train)
    with open(os.path.join(PROPHET_MODELS_PATH, files.PROPHET_2_YEARS_WEATHER_MODEL), "wb") as file:
        pickle.dump(model_energy_with_weather, file)
    logging.info("Training Prophet model on 2 years took %.2f seconds." % (time() - start_time))
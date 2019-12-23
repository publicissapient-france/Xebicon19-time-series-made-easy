import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone as tz
import pickle

import src.constants.columns as c
import src.constants.files as files
import src.constants.models as md

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_meteo_data():
    logging.info("Preprocessing meteo data.")
    main_meteo = (pd.read_csv(files.MAIN_METEO, parse_dates=[c.Meteo.DATE])
        .rename(columns={"MAX_TEMP": c.Meteo.MAX_TEMP_PARIS})[[c.Meteo.DATE, c.Meteo.MAX_TEMP_PARIS]])
    last_meteo_paris = (pd.read_csv(files.LAST_METEO_PARIS, sep=";", parse_dates=[c.Meteo.DATE])
                        .rename(columns={"MAX_TEMP": c.Meteo.MAX_TEMP_PARIS}))
    full_meteo = pd.concat([main_meteo, last_meteo_paris], axis=0)

    full_meteo.to_csv(files.FULL_METEO)


def preprocess_energy_consumption_data():
    logging.info("Reading energy consumption raw data.")
    df = (pd.read_csv(
        files.ENERGY_CONSUMPTION, sep=";", parse_dates=[c.EnergyConso.TIMESTAMP],
        usecols=[c.EnergyConso.REGION, c.EnergyConso.TIMESTAMP, c.EnergyConso.CONSUMPTION])
        .sort_values(by=[c.EnergyConso.REGION, c.EnergyConso.TIMESTAMP])
    )

    logging.info("Aggregating by hour.")
    hourly_df = format_energy_conso_by_hour(df)

    logging.info("Merging with temperature data.")
    full_meteo = pd.read_csv(files.FULL_METEO, parse_dates=[c.Meteo.DATE])
    temp_df = merge_energy_and_meteo_dfs(full_meteo, hourly_df)

    temp_df = temp_df[temp_df[c.EnergyConso.DATE_HEURE] >= md.START_TRAIN_DATE]

    logging.info("Building region df dict.")
    region_df_dict = build_region_df_dict_from_temp_df(temp_df)

    with open(files.REGION_DF_DICT, "wb") as file:
        pickle.dump(region_df_dict, file)

    return region_df_dict


def format_energy_conso_by_hour(df):
    df[c.EnergyConso.CONSUMPTION] = df[c.EnergyConso.CONSUMPTION].fillna(df[c.EnergyConso.CONSUMPTION].mean())
    df[c.EnergyConso.DATE_HEURE] = df[c.EnergyConso.TIMESTAMP].apply(half_hour_french_timestamp_to_utc_hourly_timestamp)

    hourly_df = (df
                 .groupby([c.EnergyConso.REGION, c.EnergyConso.DATE_HEURE], as_index=False)
                 .agg({c.EnergyConso.CONSUMPTION: np.sum})
                 )

    return hourly_df


def half_hour_french_timestamp_to_utc_hourly_timestamp(timestamp):
    hourly_french_timestamp = timestamp + timedelta(minutes=timestamp.minute)
    utc_hourly_timestamp = hourly_french_timestamp.astimezone(tz("UTC"))

    return datetime.fromtimestamp(utc_hourly_timestamp.timestamp())


def merge_energy_and_meteo_dfs(full_meteo, hourly_df):
    full_meteo[c.Meteo.DATE] = full_meteo[c.Meteo.DATE].apply(lambda x: x.date())

    hourly_df[c.Meteo.DATE] = hourly_df[c.EnergyConso.DATE_HEURE].apply(lambda x: x.date())

    temp_df = (pd.merge(hourly_df, full_meteo[[c.Meteo.DATE, c.Meteo.MAX_TEMP_PARIS]], on=c.Meteo.DATE, how="left")
               .drop(c.Meteo.DATE, axis=1))

    return temp_df


def build_region_df_dict_from_temp_df(temp_df):
    region_df_dict = {}

    for region in pd.unique(temp_df[c.EnergyConso.REGION]):
        region_df_dict[region] = temp_df[temp_df[c.EnergyConso.REGION] == region].copy().reset_index(drop=True)
        region_df_dict[region].index = region_df_dict[region][c.EnergyConso.DATE_HEURE]
        new_date_index = pd.date_range(start=temp_df[c.EnergyConso.DATE_HEURE].min(),
                                       end=temp_df[c.EnergyConso.DATE_HEURE].max(), freq=md.FREQ)
        region_df_dict[region] = region_df_dict[region].reindex(new_date_index).drop(
            [c.EnergyConso.DATE_HEURE, c.EnergyConso.REGION], axis=1)
        region_df_dict[region][c.Meteo.MAX_TEMP_PARIS] = region_df_dict[region][c.Meteo.MAX_TEMP_PARIS].fillna(
            method="ffill")

    return region_df_dict

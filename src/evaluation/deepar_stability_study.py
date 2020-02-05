import os
import pickle
import logging
import pandas as pd

from pathlib import Path

import src.constants.files as files
import src.constants.models as md

from src.evaluation.evaluation import prepare_data_for_deepar_plot
from src.evaluation.deepar_plots import plot_forecasts

STABILITY_STUDY_PATH = files.create_folder(
    os.path.join(files.OUTPUT_DATA, f"deepar_stability_study{files.TEST_SUFFIX}"))

STABILITY_STUDY_PLOTS = files.create_folder(os.path.join(STABILITY_STUDY_PATH, "plots"))

DEEPAR_STABILITY_STUDY_RESULTS = os.path.join(STABILITY_STUDY_PATH, "stability_study_results.xlsx")


def run_stability_study():
    region_df_dict = pickle.load(open(files.REGION_DF_DICT, "rb"))
    stability_study_results = []

    for max_epochs in eval(md.DEEPAR_MAX_EPOCHS_LIST_STR):
        logging.info(f"Running prediction 10 times with model trained with {max_epochs} epochs.")
        for trial_nb in range(1, 11):
            forecasts, tss, model_pkl_path = prepare_data_for_deepar_plot(
                region_df_dict, [md.IDF], None, max_epochs, md.LEARNING_RATE, trial_nb)

            fig_path = os.path.join(STABILITY_STUDY_PLOTS, f"{Path(model_pkl_path).name}.png")
            mape = plot_forecasts(
                region_df_dict, md.END_TRAIN_DATE, tss, forecasts, past_length=2 * md.NB_HOURS_PRED, fig_path=fig_path)
            result_dict = {"learning_rate": md.LEARNING_RATE,
                           "max_epochs": max_epochs,
                           "trial_nb": trial_nb,
                           "prediction_date": md.END_TRAIN_DATE,
                           "nb_days_pred": int(md.NB_HOURS_PRED / 24),
                           "MAPE": mape}

            stability_study_results.append(result_dict)

    stability_study_results_df = pd.DataFrame.from_records(stability_study_results)

    stability_study_results_df.to_excel(DEEPAR_STABILITY_STUDY_RESULTS, index=False)

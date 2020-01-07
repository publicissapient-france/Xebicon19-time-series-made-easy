from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions

from datetime import datetime, timedelta
import pickle
import os

import src.constants.models as md
import src.constants.files as files

import logging


DEEPAR_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = files.create_folder(os.path.join(DEEPAR_PATH, "models"))


def train_predictor(region_df_dict, end_train_date, regions_list, max_epochs, learning_rate, target_col,
                    feat_dynamic_cols=None):

    estimator = DeepAREstimator(freq=md.FREQ,
                                prediction_length=md.NB_HOURS_PRED,
                                trainer=Trainer(epochs=max_epochs, learning_rate=learning_rate,
                                                learning_rate_decay_factor=md.LR_DECAY_FACTOR),
                                use_feat_dynamic_real=feat_dynamic_cols is not None)
    if feat_dynamic_cols is not None:

        training_data = ListDataset(
            [{"item_id": region,
              "start": region_df_dict[region].index[0],
              "target": region_df_dict[region][target_col][:end_train_date],
              "feat_dynamic_real": [region_df_dict[region][feat_dynamic_col][:end_train_date]
                                    for feat_dynamic_col in feat_dynamic_cols]
              }
             for region in regions_list],
            freq=md.FREQ
        )
    else:
        training_data = ListDataset(
            [{"item_id": region,
              "start": region_df_dict[region].index[0],
              "target": region_df_dict[region][target_col][:end_train_date]
              }
             for region in regions_list],
            freq=md.FREQ
        )

    predictor = estimator.train(training_data=training_data)

    logging.info("Saving model with {} epochs and learning rate of {}".format(max_epochs, learning_rate))
    with open(predictor_path(region_df_dict, regions_list, max_epochs, learning_rate, feat_dynamic_cols), "wb") as file:
        pickle.dump(region_df_dict, file)

    return predictor


def predictor_path(region_df_dict, regions_list, max_epochs, learning_rate, feat_dynamic_cols):
    if len(regions_list) == 1:
        current_predictor_name = "{}_{}_epochs_lr_{}".format(regions_list[0], max_epochs, learning_rate)
    elif len(regions_list) == len(list(region_df_dict.keys())):
        current_predictor_name = "all_regions_{}_epochs_lr_{}".format(max_epochs, learning_rate)
    else:
        raise ValueError("You have to complexify the model naming system if you want to train the model on an "
        "incomplete subset of regions.")
    if feat_dynamic_cols is not None:
        current_predictor_name += "_" + "_".join(feat_dynamic_cols)
    # Add trial number
    existing_models = os.listdir(MODELS_PATH)
    old_trials_for_same_predictor = [model for model in existing_models if model.startswith(current_predictor_name)]
    current_predictor_name = current_predictor_name + "_trial_{}".format(len(old_trials_for_same_predictor) + 1)
    return os.path.join(MODELS_PATH, current_predictor_name)


def make_predictions(predictor, region_df_dict, test_date, regions_list, target_col, feat_dynamic_cols=None):
    if feat_dynamic_cols is not None:
        test_data = ListDataset(
            [{"item_id": region,
              "start": region_df_dict[region].index[0],
              "target": region_df_dict[region][target_col][:test_date + timedelta(hours=md.NB_HOURS_PRED)],
              "feat_dynamic_real": [region_df_dict[region][feat_dynamic_col][:test_date + timedelta(hours=md.NB_HOURS_PRED)]
                                    for feat_dynamic_col in feat_dynamic_cols]
              }
             for region in regions_list],
            freq=md.FREQ
        )
    else:
        test_data = ListDataset(
            [{"item_id": region,
              "start": region_df_dict[region].index[0],
              "target": region_df_dict[region][target_col][:test_date + timedelta(hours=md.NB_HOURS_PRED)],
              }
             for region in regions_list],
            freq=md.FREQ
        )

    forecast_it, ts_it = make_evaluation_predictions(test_data, predictor=predictor, num_eval_samples=100)

    return list(forecast_it), list(ts_it)



from main import main, TDV
from tests.prepare_raw_test_data import prepare_raw_test_data


def test_main():
    bool_dict = {"preprocess_data": False,
                 "train_deepar": False,
                 "train_prophet": False,
                 "multiple_deepar_trainings": True,
                 "run_deepar_stability_study": False,
                 "run_arima_training": False,
                 "evaluate": False}
    prepare_raw_test_data()
    main(bool_dict, TDV)

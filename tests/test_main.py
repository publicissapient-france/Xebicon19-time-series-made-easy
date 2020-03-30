from main import main, TDV
from tests.prepare_raw_test_data import prepare_raw_test_data


def test_main():
    bool_dict = {"preprocess_data": True,
                 "train_deepar": True,
                 "train_prophet": True,
                 "multiple_deepar_trainings": True,
                 "run_deepar_stability_study": True,
                 "run_arima_training": True,
                 "evaluate": True}
    prepare_raw_test_data()
    main(bool_dict, TDV)

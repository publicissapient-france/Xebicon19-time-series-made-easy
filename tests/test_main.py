from main import main, TDV


def test_main():
    bool_dict = {"preprocess_data": True,
                 "train_deepar": True,
                 "train_prophet": True,
                 "multiple_deepar_trainings": True,
                 "run_deepar_stability_study": True,
                 "run_arima_training": True,
                 "evaluate": True}
    main(bool_dict, TDV)

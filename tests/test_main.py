from main import main


def test_main():
    bool_dict = {"preprocess_data": False,
                 "train_deepar": True,
                 "train_prophet": False,
                 "evaluate": False,
                 "multiple_deepar_trainings": False}
    main(bool_dict)

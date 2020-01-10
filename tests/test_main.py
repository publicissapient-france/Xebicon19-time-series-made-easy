from main import main


def test_main():
    bool_dict = {"preprocess_data": False,
                 "train_deepar": False,
                 "train_prophet": False,
                 "evaluate": True}
    main(bool_dict)

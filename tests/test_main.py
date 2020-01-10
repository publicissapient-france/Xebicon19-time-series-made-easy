from main import main


def test_main():
    bool_dict = {"preprocess_data": False,
                 "train_deepar": True,
                 "train_prophet": True,
                 "evaluate": True}
    main(bool_dict)

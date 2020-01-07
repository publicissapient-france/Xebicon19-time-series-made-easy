from main import main


def test_main():
    bool_dict = {"preprocess_data": False,
                 "train_deepar": True}
    main(bool_dict)

from main import main


def test_main():
    bool_dict = {"preprocess_data": True,
                 "train_deepar": True}
    main(bool_dict)

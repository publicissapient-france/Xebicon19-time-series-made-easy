from src.preprocess import preprocess_meteo_data, preprocess_energy_consumption_data

def main(bool_dict):
    if bool_dict["preprocess_data"]:
        preprocess_meteo_data()
        preprocess_energy_consumption_data()


if __name__ == "__main__":
    bool_dict = {"preprocess_data": True}
    main(bool_dict)
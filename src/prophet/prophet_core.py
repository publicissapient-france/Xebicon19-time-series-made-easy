import src.constants.columns as c


def format_training_data(df_idf, start_train_date, end_train_date):
    df_idf_train = df_idf[(df_idf.index >= start_train_date) & (df_idf.index < end_train_date)].copy()

    df_idf_train["ds"] = df_idf_train.index
    df_idf_train.rename(columns={c.EnergyConso.CONSUMPTION: "y"}, inplace=True)

    df_prophet_train = df_idf_train.reset_index(drop=True)

    return df_prophet_train

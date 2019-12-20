import os

NROWS = int(os.getenv("NROWS", 10**9)) # takes value specified in pytest.ini file when launching tests.


class EnergyConso:
    TIMESTAMP = "Date - Heure"

    REGION = "Région"

    CONSUMPTION = "Consommation (MW)"

    DATE_HEURE = "date_heure"


class Meteo:
    DATE = "DATE"
    MAX_TEMP_PARIS = "max_temp_paris"

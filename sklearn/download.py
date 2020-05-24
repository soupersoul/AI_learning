import os
import requests

DOWNLOAD_ROOT="https://raw.githubusercontent.com/soupersoul/handon-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.csv"

def download_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok = True)
    csv_path = os.path.join(housing_path, "housing.csv")
    r = requests.get(housing_url)
    with open(csv_path, "wb") as f : 
        f.write(r.content)
    r.close()

download_data()
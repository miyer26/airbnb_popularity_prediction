# author: Mukund Iyer
# date: 12/28/21

"""
This script will create new features for the data set and complete the preprocessing 
of the data.

Usage: preprocessing.py --train_data_path=<train_data_path> --test_data_path=<test_data_path> --results_folder_path=<results_path> 

Options: 
--train_data_path=<train_data_path>   The path to the processed train split of the AirBnB data
--test_data_path=<test_data_path>     The path to the processed test split of the AirBnB data
--results_folder_path=<results_path>         The path where the results of the preprocessing are saved

"""
from docopt import docopt
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
import datetime
from tabulate import tabulate
import pickle
import os

from sklearn.metrics import make_scorer

opt = docopt(__doc__)


def create_interval(X_train, X_test):
    # engineering new 'interval' feature
    X_train["last_review"] = pd.to_datetime(X_train["last_review"])
    X_test["last_review"] = pd.to_datetime(X_test["last_review"])

    X_train["interval"] = X_train.apply(
        lambda x: (
            (datetime.datetime(2019, 12, 31) - x.last_review) / (np.timedelta64(1, "D"))
        ),
        axis=1,
    )
    X_test["interval"] = X_test.apply(
        lambda x: (
            (datetime.datetime(2019, 12, 31) - x.last_review) / (np.timedelta64(1, "D"))
        ),
        axis=1,
    )

    print("Interval feature created")


def create_prop_owned(X_train, X_test):
    X_train["host_id_count"] = X_train["host_id"].map(X_train["host_id"].value_counts())
    X_test["host_id_count"] = X_test["host_id"].map(X_test["host_id"].value_counts())

    conditions = [
        (X_train["host_id_count"] == 1),
        (X_train["host_id_count"] > 1 & (X_train["host_id_count"] <= 5)),
        (X_train["host_id_count"] >= 5),
    ]
    values = ["single", "few", "many"]
    X_train["properties_owned"] = np.select(conditions, values)

    conditions = [
        (X_test["host_id_count"] == 1),
        (X_test["host_id_count"] > 1 & (X_test["host_id_count"] <= 5)),
        (X_test["host_id_count"] >= 5),
    ]
    values = ["single", "few", "many"]
    X_test["properties_owned"] = np.select(conditions, values)

    print("Property owner feature created")


def create_preprocessor():
    properties_owned_levels = ["single", "few", "many"]

    numeric_features = [
        "latitude",
        "longitude",
        "price",
        "minimum_nights",
        "calculated_host_listings_count",
        "availability_365",
        "interval",
    ]

    categoric_features1 = ["neighbourhood_group", "room_type", "neighbourhood"]
    categoric_features2 = "description"
    ordinal_features = ["properties_owned"]
    drop = [
        "id",
        "host_name",
        "number_of_reviews",
        "last_review",
        "host_id",
        "host_id_count",
    ]

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore", dtype=int), categoric_features1),
        (CountVectorizer(stop_words="english"), categoric_features2),
        (
            OrdinalEncoder(categories=[properties_owned_levels], dtype=int),
            ordinal_features,
        ),
        ("drop", drop),
    )

    print("Created preprocessor")

    return preprocessor


def main(train_data_path, test_data_path, results_path):

    # reading train data csv
    train_data = pd.read_csv(train_data_path, index_col=0)
    test_data = pd.read_csv(test_data_path, index_col=0)

    # creating X_train, y_train from test and train dataframes
    X_train = train_data.drop(columns=["reviews_per_month"])
    y_train = train_data["reviews_per_month"]
    X_test = test_data.drop(columns=["reviews_per_month"])
    y_test = test_data["reviews_per_month"]

    # engineering new 'interval' feature
    create_interval(X_train, X_test)

    # engineering new 'properties_owned' feature
    create_prop_owned(X_train, X_test)

    # saving X_train and y_train with engineered features
    processed_path = (
        test_data_path.split("/", 3)[0:3][0]
        + "/"
        + test_data_path.split("/", 3)[0:3][1]
        + "/"
        + test_data_path.split("/", 3)[0:3][2]
    )
    X_train.to_csv(processed_path + "/X_train.csv")
    y_train.to_csv(processed_path + "/y_train.csv")
    X_test.to_csv(processed_path + "/X_test.csv")
    y_test.to_csv(processed_path + "/y_test.csv")

    # preprocessing
    preprocessor = create_preprocessor()

    # create preprocessing folder in results folder
    if not os.path.exists(results_path + "/preprocessing"):
        os.makedirs(results_path + "/preprocessing")

    # saving preprocessor object
    with open(f"{results_path}/preprocessing/preprocessor.pickle", "wb") as f:
        pickle.dump(preprocessor, f)


if __name__ == "__main__":
    main(
        opt["--train_data_path"], opt["--test_data_path"], opt["--results_folder_path"]
    )

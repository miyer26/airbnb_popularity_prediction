# author: Mukund Iyer
# date: 2021-12-27

"""
This script takes the raw data from the csv file, cleans it and converts 
the data into a train and test split.

Usage: clean_data.py --raw_data_path=<file_path> --train_data_path=<train_data_path> --test_data_path=<test_data_path> 

Options: 
--raw_data_path=<file_path>           The path to the raw AirBnB data 
--train_data_path=<test_data_path>    The full path to the processed train split csv of the AirBnB data
--test_data_path=<train_data_path>    The full path to the processed test split csv of the AirBnB data
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from docopt import docopt

opt = docopt(__doc__)


def main(raw_path, train_path, test_path):

    data = pd.read_csv(raw_path)

    # renaming 'name' column to 'description'
    data = data.rename(columns={"name": "description"})

    # fill the NAs in the 'description' column
    data["description"] = data["description"].fillna("?")

    # drop the NAs in the traget column 'reviews_per_month'
    data = data.dropna(subset=["reviews_per_month"])

    # split the data into train and test
    train_df, test_df = train_test_split(data, test_size=0.9, random_state=123)

    shape_train = train_df.shape
    shape_test = test_df.shape
    print(f"The dimensions of the train data is {shape_train}")
    print(f"The dimensions of the test data is {shape_test}")

    # creating the processed data file path for the train data
    path_check_train = (
        train_path.split("/", 3)[0:3][0]
        + "/"
        + train_path.split("/", 3)[0:3][1]
        + "/"
        + train_path.split("/", 3)[0:3][2]
    )
    if not os.path.exists(path_check_train):
        os.makedirs(path_check_train)

    # creating the processed data file path for the test data
    path_check_test = (
        test_path.split("/", 3)[0:3][0]
        + "/"
        + test_path.split("/", 3)[0:3][1]
        + "/"
        + test_path.split("/", 3)[0:3][2]
    )
    if not os.path.exists(path_check_test):
        os.makedirs(path_check_test)

    # saving the processed data files
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)


if __name__ == "__main__":
    main(opt["--raw_data_path"], opt["--train_data_path"], opt["--test_data_path"])

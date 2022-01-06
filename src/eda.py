# author: Mukund Iyer
# date: 2021-12-28

"""
This script completes the exploratory data analysis using the training data and produces 
pngs of the plots.

Usage: eda.py --train_data_path=<train_data_path> --results_folder_path=<results_folder_path> 

Options: 
--train_data_path=<train_data_path>   The path to the processed train split of the AirBnB data
--results_folder_path=<results_path>  The path to the folder where the plots from the EDA are saved 
"""

import os
from docopt import docopt
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from altair_saver import save
import pickle
import seaborn as sns

opt = docopt(__doc__)


def hist_feat(train_data, results_folder_path):

    numeric_features_and_target = [
        "latitude",
        "longitude",
        "price",
        "minimum_nights",
        "calculated_host_listings_count",
        "availability_365",
        "reviews_per_month",
    ]

    # histogram for each feature
    for feature in numeric_features_and_target:
        feat = feature
        ax = train_data[feature].plot.hist(bins=20, alpha=0.5, legend=True)
        plt.xlabel(feature)
        plt.title("Histogram of " + feature)
        file_name = feature.strip()
        plt.tight_layout()
        plt.savefig(f"{results_folder_path}/eda/hist_{file_name}.png")
        plt.clf()

    print("Histograms have been created")


def corr_plot(train_data, results_folder_path):

    numeric_features_and_target = [
        "latitude",
        "longitude",
        "price",
        "minimum_nights",
        "calculated_host_listings_count",
        "availability_365",
        "reviews_per_month",
    ]

    corr = train_data[numeric_features_and_target].corr()
    corr_plot = sns.heatmap(corr, annot=True, cmap=plt.cm.Blues)
    corr_plot.figure.savefig(
        f"{results_folder_path}/eda/correlation.png", bbox_inches="tight"
    )

    print("Correlation plot created")


def scatter_plots(train_data, results_folder_path):
    # scatter latitude/longtitude
    scatter_long_lat = (
        alt.Chart(train_data)
        .mark_circle()
        .encode(
            alt.X(alt.repeat(), type="quantitative", title="Latitude"),
            alt.Y("reviews_per_month"),
        )
        .properties(width=300, height=200)
        .repeat(["latitude", "longitude"])
    )
    scatter_long_lat.save(f"{results_folder_path}/eda/scatter_long_lat.png")

    # scatter price/minimum nights
    alt.Chart(train_data).mark_point().encode(
        alt.X(alt.repeat(), type="quantitative"),
        alt.Y("reviews_per_month"),
    ).properties(width=300, height=200).repeat(["price", "minimum_nights"])
    scatter_long_lat.save(f"{results_folder_path}/eda/scatter_price_min_nights.png")

    # scatter host listings/availability365
    alt.Chart(train_data).mark_point().encode(
        alt.X(alt.repeat(), type="quantitative"),
        alt.Y("reviews_per_month"),
    ).properties(width=300, height=200).repeat(
        ["calculated_host_listings_count", "availability_365"]
    )
    scatter_long_lat.save(
        f"{results_folder_path}/eda/scatter_hostlistings_availability.png"
    )

    print("Scatter plots have been created")


def main(train_path, results_folder_path):

    # checking the input for the path and importing the csv
    assert train_path.endswith(".csv"), "Input should be a .csv file as the <in_file>"

    train_data = pd.read_csv(train_path)

    # creating the processed data file path for the test data
    path_check_test = results_folder_path + "/eda"

    if not os.path.exists(path_check_test):
        os.makedirs(path_check_test)

    # histograms
    hist_feat(train_data, results_folder_path)

    # scatter plots
    scatter_plots(train_data, results_folder_path)

    # correlation plot
    corr_plot(train_data, results_folder_path)

    return


if __name__ == "__main__":
    main(opt["--train_data_path"], opt["--results_folder_path"])

# author: Mukund Iyer
# date: 2021-12-28

"""
This script completes the exploratory data analysis using the training data and produces 
pngs of the plots.

Usage: eda.py --train_data_path=<train_data_path> --results_path=<results_path> 

Options: 
--train_data_path=<train_data_path>   The path to the processed train split of the AirBnB data
--results_path=<results_path>         The path where the plots from the EDA are saved 
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


def main(train_path, results_path):

    assert train_path.endswith(".csv"), "Input should be a .csv file as the <in_file>"

    train_data = pd.read_csv(train_path)

    # histograms
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
        plt.savefig(f"{results_path}/hist_{file_name}.png")
        print(f"{results_path}/hist_{file_name}.png")
        plt.clf()

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
    scatter_long_lat.save(f"{results_path}/scatter_long_lat.png")

    # scatter price/minimum nights
    alt.Chart(train_data).mark_point().encode(
        alt.X(alt.repeat(), type="quantitative"),
        alt.Y("reviews_per_month"),
    ).properties(width=300, height=200).repeat(["price", "minimum_nights"])
    scatter_long_lat.save(f"{results_path}/scatter_price_min_nights.png")

    # scatter host listings/availability365
    alt.Chart(train_data).mark_point().encode(
        alt.X(alt.repeat(), type="quantitative"),
        alt.Y("reviews_per_month"),
    ).properties(width=300, height=200).repeat(
        ["calculated_host_listings_count", "availability_365"]
    )
    scatter_long_lat.save(f"{results_path}/scatter_hostlistings_availability.png")

    # correlation plot
    corr = train_data[numeric_features_and_target].corr()
    corr_plot = sns.heatmap(corr, annot=True, cmap=plt.cm.Blues)
    corr_plot.figure.savefig(f"{results_path}/correlation.png", bbox_inches="tight")


if __name__ == "__main__":
    main(opt["--train_data_path"], opt["--results_path"])

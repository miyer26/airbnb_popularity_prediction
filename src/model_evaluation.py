# author: Mukund Iyer
# date: 12/29/21

"""
This script aims to evaluate the output of the best model and determine its most important features through SHAPing.

Usage: model_evaluation.py --processed_data_path=<processed_data_path> --results_path=<results_path> 

Options: 
--processed_data_path=<processed_data_path>   The path to the processed data folder
--results_path=<results_path>         The path where the results of the preprocessing are saved

"""

import shap
import pickle
from docopt import docopt
import os
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error


opt = docopt(__doc__)


def load_preproc(results_path):
    with open(f"{results_path}/preprocessor.pickle", "rb") as f:
        preprocessor = pickle.load(f)
    return preprocessor


def feature_names(preprocessor, X_train):

    numeric_features = [
        "latitude",
        "longitude",
        "price",
        "minimum_nights",
        "calculated_host_listings_count",
        "availability_365",
        "interval",
    ]

    ordinal_features = ["properties_owned"]

    preprocessor.fit(X_train)

    column_names = (
        numeric_features
        + list(
            preprocessor.named_transformers_["onehotencoder"].get_feature_names_out()
        )
        + list(
            preprocessor.named_transformers_["countvectorizer"].get_feature_names_out()
        )
        + ordinal_features
    )

    return column_names


def final_results(model, X_test, y_test):

    test_results = {}

    test_results["Rsquared"] = model.score(X_test, y_test)
    test_results["MAPE"] = mean_absolute_percentage_error(
        y_test,
        pipe_LGBMR.predict(X_test),
    )
    test_results["RMSE"] = mean_squared_error(
        y_test, pipe_LGBMR.predict(X_test), squared=False
    )

    return test_results


def main(processed_data_path, results_path):

    # loading preprocessor object
    preprocessor = load_preproc(results_path)

    # loading X_train and y_train
    X_train = pd.read_csv(f"{processed_data_path}/X_train.csv", index_col=0)
    y_train = pd.read_csv(f"{processed_data_path}/y_train.csv", index_col=0)

    column_names = feature_names(preprocessor, X_train)

    X_train_proc = pd.DataFrame(
        data=preprocessor.transform(X_train).toarray(),
        columns=column_names,
        index=X_train.index,
    )

    pipe_LGBMR = make_pipeline(preprocessor, LGBMRegressor(random_state=123))
    pipe_LGBMR.fit(X_train, y_train.values.ravel())

    lgbm_explainer = shap.TreeExplainer(pipe_LGBMR.named_steps["lgbmregressor"])
    train_lgbm_shap_values = lgbm_explainer.shap_values(X_train_proc)

    pipe_LGBMR.fit(X_train, y_train)

    # global feature importance plot
    shap.summary_plot(
        train_lgbm_shap_values,
        X_train_proc,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{results_path}/feat_imp_global.png")
    print(f"{results_path}/feat_imp_global.png")
    plt.clf()

    # feature importance and magnitude
    shap.summary_plot(train_lgbm_shap_values, X_train_proc, show=False)
    plt.tight_layout()
    plt.savefig(f"{results_path}/feat_imp_mag_dir.png")
    print(f"{results_path}/feat_imp_mag_dir.png")
    plt.clf()

    # loading X_test and y_test
    X_test = pd.read_csv(f"{processed_data_path}/X_test.csv", index_col=0)
    y_test = pd.read_csv(f"{processed_data_path}/y_test.csv", index_col=0)

    # test scores
    test_score_dict = final_results(pipe_LGBMR, X_test, y_test)

    print(test_score_dict)


if __name__ == "__main__":
    main(opt["--processed_data_path"], opt["--results_path"])

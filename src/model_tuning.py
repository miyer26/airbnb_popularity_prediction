# author: Mukund Iyer
# date: 12/29/21

"""
This script will carry out cross-validation and hyperparameter optimization for different models
of the data.

Usage: model_tuning.py --processed_data_path=<processed_data_path> --results_path=<results_path> 

Options: 
--processed_data_path=<processed_data_path>   The path to the processed data folder
--results_path=<results_path>         The path where the results of the preprocessing are saved

"""


from pandas.io.parsers import read_csv
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    train_test_split,
)
import pickle
from docopt import docopt
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

opt = docopt(__doc__)

# function to load preprocessor
def load_preproc(results_path):
    with open(f"{results_path}/preprocessor.pickle", "rb") as f:
        preprocessor = pickle.load(f)
    return preprocessor


# function to define MAPE score
def mape(true, pred):
    return 100.0 * np.mean(np.abs((pred - true) / true))


def scoring():

    # make a scorer function that we can pass into cross-validation
    mape_scorer = make_scorer(mape, greater_is_better=False)

    scoring_metrics = {
        "neg RMSE": "neg_root_mean_squared_error",
        "r2": "r2",
        # "mape": mape_scorer,
    }
    return scoring_metrics


# function adapted from Varada Kolhatkar
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def scoring_dict():
    return {}


# baseline model - dummy regressor
def dummy_model(preprocessor, X_train, y_train, scoring_metrics, results_dict):

    pipe_dummy = make_pipeline(preprocessor, DummyRegressor())
    results_dict["Dummy"] = mean_std_cross_val_scores(
        pipe_dummy,
        X_train,
        y_train,
        return_train_score="True",
        scoring=scoring_metrics,
        n_jobs=-1,
    )

    print(results_dict)


def tuned_ridge_model(preprocessor, X_train, y_train, scoring_metrics, results_dict):

    pipe_ridge = make_pipeline(preprocessor, Ridge())

    param_grid = {
        "columntransformer__countvectorizer__max_features": [
            200,
            500,
            1000,
            1500,
            2000,
        ],
        "ridge__alpha": [5, 10, 20, 50, 100, 125],
    }

    #
    random_search_ridge = RandomizedSearchCV(
        pipe_ridge,
        param_distributions=param_grid,
        n_jobs=-1,
        n_iter=8,
        cv=3,
        random_state=123,
        return_train_score=True,
    )

    random_search_ridge.fit(X_train, y_train)

    # CV on best model
    results_dict["Ridge"] = mean_std_cross_val_scores(
        random_search_ridge,
        X_train,
        y_train,
        return_train_score="True",
        scoring=scoring_metrics,
        n_jobs=-1,
    )

    print(results_dict)


def main(processed_data_path, results_path):

    # loading preprocessor object
    preprocessor = load_preproc(results_path)

    # loading X_train and y_train
    X_train = pd.read_csv(f"{processed_data_path}/X_train.csv", index_col=0)
    y_train = pd.read_csv(f"{processed_data_path}/y_train.csv", index_col=0)

    scoring_metrics = scoring()

    results = scoring_dict()

    dummy_model(preprocessor, X_train, y_train, scoring_metrics, results)

    tuned_ridge_model(preprocessor, X_train, y_train, scoring_metrics, results)


if __name__ == "__main__":
    main(opt["--processed_data_path"], opt["--results_path"])

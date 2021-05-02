from numpy import array, mean, cov, linalg, where
from scipy.stats import chi2
import matplotlib.pyplot as plt
from SQL.NewTablesScripts.constants import *


def mahalanobis(x=None, data=None):
    """
    Function that computes the Mahalanobis Distance between each row of x and the data

    :param x: One item of the dataframe.
    :param data: Dataframe from which Mahalanobis distance of each observation of x is to be computed.
    """
    # We transform the data to numpy array
    data = data.to_numpy()

    # We get the center point, the vector composed of the means of each variable
    centerpoint = mean(data, axis=0)

    # We get the covariance matrix
    covariance = cov(data, rowvar=False)

    # We get the inverse of the covariance matrix
    covariance_inv = linalg.matrix_power(covariance, -1)

    return (x - centerpoint).T.dot(covariance_inv).dot(x - centerpoint)


def get_outlier_ids(df, cont_cols, cat_cols=None):
    """
    Function that returns the IDs of the rows detected as outliers

    :param df: the dataframe containing the data
    :param cont_cols: the continuous columns in this dataframe
    :param cat_cols: the categorical columns in this dataframe
    """

    if cat_cols is None:
        data = df[cont_cols]

    # We prepare the data the detection of the outliers
    data = data.astype("float").fillna(data.mean())

    # We calculate the mahalanobis distance of each point
    distances = [mahalanobis(x, data) for i, x in enumerate(data.to_numpy())]

    # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers
    cutoff = chi2.ppf(0.95, data.shape[1])

    # We get the index of the outliers
    outlier_index = where(distances > cutoff)[0].tolist()

    # We get IDs of the patients detected as outliers
    IDs = [df[PARTICIPANT][index] for index in outlier_index]

    return IDs



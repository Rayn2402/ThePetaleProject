"""
This file stores the class DataCleaner used remove invalid rows and columns from learning dataframes
"""

import pandas as pd
from numpy import mean, cov, linalg, array, einsum, sqrt
from scipy.stats import chi2
from SQL.NewTablesScripts.constants import PARTICIPANT
from SQL.DataManager.Helpers import retrieve_numerical
from typing import List, Optional
from json import dump


class DataCleaner:
    """
    Object use to clean dataframes used for our learning task.

    It removes rows and columns with too many missing values (percentage above thresh)
    and then remove outliers according to their numerical attributes.
    """
    # RECORDS CSTS
    CP = "Critical Patients"
    CC = "Critical Columns"
    CQD = "Column Quartile Details"

    # PRE-SAVED WARNING MESSAGES
    ROW_TRESHOLD_WARNING = "Row threshold condition not fulfilled"
    COL_THRESHOLD_WARNING = "Column threshold condition not fulfilled"

    # OUTLIER LVLS
    LOW = "LOW"
    HIGH = "HIGH"

    # MAHALANOBIS COLUMN ID
    MAHALANOBIS = "Mahalanobis"

    def __init__(self, column_thresh: float, row_thresh: float,
                 outlier_alpha: float, qchi2_mahalanobis_cutoff: float, records_path: str):
        """
        Class constructor

        :param column_thresh: percentage threshold (0 <= thresh <= 1)
        :param row_thresh: percentage threshold (0 <= thresh <= 1)
        :param outlier_alpha: constant multiplied by inter quartile range (IQR) to determine outliers
        :param qchi2_mahalanobis_cutoff: Chi-squared quantile used to determine Mahalanobis cutoff value
        :param records_path: json file path to save results of cleaning
        """
        # Internal private fixed attributes
        self.__column_tresh = column_thresh
        self.__row_thresh = row_thresh
        self.__outlier_alpha = outlier_alpha
        self.__qchi2 = qchi2_mahalanobis_cutoff
        self.__records_path = records_path

        # Private mutable attribute
        self.__records = {self.CP: {},
                          self.CC: [],
                          "Column Threshold": self.__column_tresh,
                          "Row Threshold": self.__row_thresh,
                          "Outlier Alpha": self.__outlier_alpha,
                          self.CQD: {},
                          self.MAHALANOBIS: {}}

    def __call__(self, df: pd.DataFrame, numerical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Executes all the cleaning steps and save records of the procedure

        :param df: pandas dataframe
        :param numerical_columns: list with names of numerical columns
        :return: cleaned pandas dataframe
        """
        # We find numerical columns if none are mentioned
        if numerical_columns is None:
            numerical_df = retrieve_numerical(df, [])
            numerical_columns = list(numerical_df.columns.values)

        # We proceed to data cleaning
        updated_df = self.__identify_critical_rows_and_columns(df)
        updated_df = self.__refactor_dataframe(updated_df, numerical_columns)
        self.__identify_univariate_outliers(updated_df, numerical_columns)
        self.__identify_multivariate_outliers(updated_df, numerical_columns)

        # We save the records
        self.__save_records()

        return updated_df

    def __identify_critical_rows_and_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows and columns with too many missing values
        """

        # Column cleaning
        updated_df = df.dropna(axis=1, thresh=round(df.shape[0]*(1-self.__column_tresh)))

        # Row cleaning
        updated_df = updated_df.dropna(axis=0, thresh=round(updated_df.shape[1]*(1-self.__row_thresh)))

        # Records update
        for participant in df[PARTICIPANT].values:
            if participant not in updated_df[PARTICIPANT].values:
                self.__records[self.CP][participant] = {0: self.ROW_TRESHOLD_WARNING}

        for column in df.columns:
            if column not in updated_df.columns:
                self.__records[self.CC][column] = self.COL_THRESHOLD_WARNING

        return updated_df

    def __identify_univariate_outliers(self, df: pd.DataFrame, numerical_columns: List[str]) -> None:
        """
        Identifies patients (rows) with numerical attribute value lower than
        Q1 - alpha*(Q3-Q1) or higher than Q3 + alpha*(Q3-Q1)

        :param df: pandas dataframe
        :param numerical_columns: list with names of numerical columns
        """
        # For each column, computes quartiles and IQR = (Q3 - Q1) and identify outliers
        for c in numerical_columns:

            # Quartiles computation
            quartiles = list((df[c].quantile([0.25, 0.5, 0.75])).values)
            q1, q2, q3 = quartiles
            iqr = round(q3 - q1, 4)
            self.__records[self.CQD][c] = {"Q1": q1, "Q2": q2, "Q3": q3, "IQR": iqr}

            # Outliers identification (< Q1)
            low_outliers = df.loc[df[c] < q1 - self.__outlier_alpha*iqr, [PARTICIPANT, c]]
            self.__update_outliers_records(low_outliers, c, self.LOW)

            # Outliers identification (> Q3)
            high_outliers = df.loc[df[c] > q3 + self.__outlier_alpha*iqr, [PARTICIPANT, c]]
            self.__update_outliers_records(high_outliers, c, self.HIGH)

    def __identify_multivariate_outliers(self, df: pd.DataFrame, numerical_columns: List[str]) -> None:
        """
        Identifies patients with Mahalanobis distances abnormally high
        (over the qchi2 quantile of Chi-squared distribution with D degrees of freedom)
        where D is the number of numerical dimensions in our dataframe (after cleaning).

        :param df: pandas dataframe
        :param numerical_columns: list with names of numerical columns
        """
        # We calculate Mahalanobis distances for all patients
        temporary_df = df.loc[:, [PARTICIPANT]]
        temporary_df[self.MAHALANOBIS] = self.__calculate_squared_mahalanobis_distances(df, numerical_columns)
        print(temporary_df.columns)

        # We calculate a cutoff values based on a Chi-squared distribution
        cutoff = chi2.ppf(self.__qchi2, len(numerical_columns))

        # We save distances to records
        for i in range(temporary_df.shape[0]):

            # We extract patient's id and value
            patient, value = temporary_df.iloc[i, 0], round(temporary_df.iloc[i, 1], 4)
            self.__records[self.MAHALANOBIS][patient] = value

            # We had a warning message to the patient if its value is over the cutoff
            if value > cutoff:
                warning = self.__return_mahalanobis_warning(value)
                self.__update_single_outlier_records(patient, warning)

    def __update_outliers_records(self, subset: pd.DataFrame,
                                  column: str, lvl: str) -> None:
        """
        Adds outliers data to records

        :param subset: subset of pandas dataframe
        :param column: name of the numerical column
        :param lvl: "Low" or "High" depending if value is lower than Q1 - alpha*(Q3-Q1) or higher than
                    Q3 + alpha*(Q3-Q1)

        """
        for i in range(subset.shape[0]):

            # We extract patient's id and value
            patient, value = subset.iloc[i, 0], subset.iloc[i, 1]

            # We create the appropriate warning message
            warning = self.__return_outlier_warning(column, value, lvl)

            # We add the warning to the records
            self.__update_single_outlier_records(patient, warning)

    def __update_single_outlier_records(self, patient: str, warning: str):
        """
        Add warning message to a patient in the records

        :param patient: str
        :param warning: str, warning message
        """
        if patient in self.__records[self.CP].keys():
            self.__records[self.CP][patient].update({len(self.__records[self.CP][patient].keys()): warning})
        else:
            self.__records[self.CP][patient] = {0: warning}

    def __save_records(self) -> None:
        """
        Saves the records dictionary into a json file
        """
        # We save all the removed rows and columns collected in a json file
        with open(self.__records_path, "w") as file:
            dump(self.__records, file, indent=True)

    @staticmethod
    def __calculate_squared_mahalanobis_distances(df: pd.DataFrame, numerical_columns: List[str]) -> array:
        """
        Computes mahalanobis distances of all patients according to their numerical values

        :param df: pandas dataframe
        :param numerical_columns: list with names of numerical columns
        :return: numpy array
        """
        # We fill NaN values with column means transform the data to numpy array
        numerical_df = df[numerical_columns].fillna(df[numerical_columns].mean())
        X = numerical_df.to_numpy().reshape((df.shape[0], -1))

        # We get the center point, the vector composed of the means of each variable
        centroid = mean(X, axis=0).reshape(1, numerical_df.shape[1])

        # We get the inverse of the covariance matrix
        covariance_inv = linalg.matrix_power(cov(X, rowvar=False), -1)

        # We compute the first matrix multiplication
        A = (X - centroid)
        B = A.dot(covariance_inv)

        # We finish calculating distances all at once with einsum
        squared_distances = einsum('ij,ij->i', B, A)

        return squared_distances

    @staticmethod
    def __refactor_dataframe(df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        """
        Filter the dataframe to only keep PARTICIPANT column and specific numerical column

        :param df: pandas dataframe
        :param numerical_columns: list with names of numerical columns
        :return: same dataframe with only numeric
        """
        assert PARTICIPANT in df.columns.values, f"{PARTICIPANT} column must be in the dataframe"
        df[numerical_columns] = df[numerical_columns].astype(float)

        return df

    @staticmethod
    def __return_outlier_warning(numerical_attribute: str, value: float, lvl: str) -> str:
        """
        Returns a string to use as warning message in the records.

        :param numerical_attribute: name of the column in the dataframe
        :param value: value of the patient
        :param lvl: "Low" or "High" depending if value is lower than Q1 - alpha*(Q3-Q1) or higher than
                    Q3 + alpha*(Q3-Q1)
        :return: str
        """
        return f"Patient targeted as an outlier" \
               f" due to a {lvl} '{numerical_attribute}' value of {value}"

    @staticmethod
    def __return_mahalanobis_warning(value: float) -> str:
        """
        Returns a string to use as warning message in the records.
        The warning is for a Mahalanobis distance that is to high
        :param value: Patient's Mahalanobis distance
        :return: str
        """
        return f"Patient targeted as an outlier due to its Mahalanobis distance of {value}"

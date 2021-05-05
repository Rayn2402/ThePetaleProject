"""
This file stores the class DataCleaner used remove invalid rows and columns from learning dataframes
"""
import pandas as pd
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
    RP = "Removed Patients"
    RC = "Removed Columns"
    CQD = "Column Quartile Details"

    # PRE-SAVED WARNING MESSAGES
    ROW_TRESHOLD_WARNING = "Row threshold condition was not fulfilled"
    COL_THRESHOLD_WARNING = "Column threshold condition was not fulfilled"

    # OUTLIER LVLS
    LOW = "LOW"
    HIGH = "HIGH"

    def __init__(self, column_thresh: float, row_thresh: float,
                 outlier_alpha: float, records_path: str):
        """
        Class constructor

        :param column_thresh: percentage threshold (0 <= thresh <= 1)
        :param row_thresh: percentage threshold (0 <= thresh <= 1)
        :param outlier_alpha: constant multiplied by inter quartile range (IQR) to determine outliers
        :param records_path: json file path to save results of cleaning
        """
        # Internal private fixed attributes
        self.__column_tresh = column_thresh
        self.__row_thresh = row_thresh
        self.__outlier_alpha = outlier_alpha
        self.__records_path = records_path

        # Private mutable attribute
        self.__records = {self.RP: {},
                          self.RC: [],
                          "Column Threshold": self.__column_tresh,
                          "Row Threshold": self.__row_thresh,
                          "Outlier Alpha": self.__outlier_alpha,
                          self.CQD: {}}

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
        updated_df = self.__clean_dataframe(df)
        updated_df = self.__refactor_dataframe(updated_df, numerical_columns)
        updated_df = self.__remove_outliers(updated_df, numerical_columns)

        # We save the records
        self.__save_records()

        return updated_df

    def __clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
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
                self.__records[self.RP][participant] = self.ROW_TRESHOLD_WARNING

        for column in df.columns:
            if column not in updated_df.columns:
                self.__records[self.RC][column] = self.COL_THRESHOLD_WARNING

        return updated_df

    def __remove_outliers(self, df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        """
        Removes patients (rows) with numerical attribute value lower than
        Q1 - alpha*(Q3-Q1) or higher than Q3 + alpha*(Q3-Q1)

        :param df: pandas dataframe
        :param numerical_columns: list with names of numerical columns
        """
        # For each column, computes quartiles and IQR = (Q3 - Q1) and remove outliers
        updated_df = df
        for c in numerical_columns:

            # Quartiles computation
            quartiles = list((df[c].quantile([0.25, 0.5, 0.75])).values)
            q1, q2, q3 = quartiles
            iqr = round(q3 - q1, 4)
            self.__records[self.CQD][c] = {"Q1": q1, "Q2": q2, "Q3": q3, "IQR": iqr}

            # Outliers removal (< Q1)
            low_outlier = df.loc[df[c] < q1 - self.__outlier_alpha*iqr, [PARTICIPANT, c]]
            updated_df = self.__update_outliers_records(df, low_outlier, c, self.LOW)

            # Outliers removal (> Q3)
            high_outlier = df.loc[df[c] > q3 + self.__outlier_alpha*iqr, [PARTICIPANT, c]]
            updated_df = self.__update_outliers_records(updated_df, high_outlier, c, self.HIGH)

        return updated_df

    def __update_outliers_records(self, complete_df: pd.DataFrame, subset: pd.DataFrame,
                                  column: str, lvl: str) -> pd.DataFrame:
        """
        Adds outliers data to records

        :param subset: subset of pandas dataframe
        :param column: name of the numerical column
        :param lvl: "Low" or "High" depending if value is lower than Q1 - alpha*(Q3-Q1) or higher than
                    Q3 + alpha*(Q3-Q1)

        """
        for i in range(subset.shape[0]):
            self.__records[self.RP][subset.iloc[i, 0]] = \
                self.__return_outlier_warning(column, subset.iloc[i, 1], lvl)

        return complete_df.loc[~complete_df[PARTICIPANT].isin(list(subset[PARTICIPANT].values))]

    def __save_records(self) -> None:
        """
        Saves the records dictionary into a json file
        """
        # We save all the removed rows and columns collected in a json file
        with open(self.__records_path, "w") as file:
            dump(self.__records, file, indent=True)

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
        return f"Patient was considered as an outlier" \
               f" due to a {lvl} '{numerical_attribute}' value of {value}"

    @staticmethod
    def __return_mahalanobis_warning(value: float) -> str:
        """
        Returns a string to use as warning message in the records.
        The warning is for a Mahalanobis distance that is to high
        :param value: Patient's Mahalanobis distance
        :return: str
        """
        return f"Patient was considered as an outlier due to its Mahalanobis distance of {value}"

"""
Filename: data_management.py

Author: Nicolas Raymond
        Mehdi Mitiche

Description: Defines the DataManager class that helps interation with SQL data

Date of last modification : 2021/11/02
"""

import csv
import psycopg2
import pandas as pd
import os

from src.data.extraction import chart_services, helpers
from src.data.extraction.constants import *
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple


class DataManager:
    """
    Object that can interact with a PostgresSQL database
    """
    def __init__(self,
                 user: str,
                 password: str,
                 database: str,
                 host: str = 'localhost',
                 port: str = '5432',
                 schema: str = 'public'):
        """
        Sets the private attributes

        Args:
            user: username to access the database
            password: password linked to the user
            database: name of the database
            host: database host address
            port: connection port number
            schema: schema we want to access in the database
        """
        self.__conn, self.__cur = DataManager.connect(user, password, database, host, port)
        self.__schema = schema
        self.__database = database

    @staticmethod
    def connect(user: str,
                password: str,
                database: str,
                host: str,
                port: str) -> Tuple[Any, Any]:
        """
        Creates a connection with a database

        Args:
            user: username to access the database
            password: password linked to the user
            database: name of the database
            host: database host address
            port: connection port number

        Returns: connection, cursor
        """
        try:
            conn = psycopg2.connect(database=database,
                                    user=user,
                                    host=host,
                                    password=password,
                                    port=port)
            cur = conn.cursor()

        except psycopg2.Error as e:
            print(e.pgerror)
            raise

        return conn, cur

    def __create_table(self,
                       table_name: str,
                       types: Dict[str, str],
                       primary_key: Optional[List[str]] = None) -> None:
        """
        Creates a table named "table_name" that has the columns with the types indicated
        in the dictionary types.

        Args:
            table_name: name of the table
            types: dictionary with names of the columns (key) and their respective types (value)
            primary_key: list of column names to use as primary key (or composite key when more than 1)

        Returns:
        """
        # We save the start of the query
        query = f"CREATE TABLE {self.__schema}.\"{table_name}\" (" + helpers.colsAndTypes(types)

        # If a primary key is given we add it to the query
        if primary_key is not None:

            # We define the primary key
            keys = helpers.colsForSql(primary_key)
            query += f", PRIMARY KEY ({keys}) );"

        else:
            query += ");"

        # We execute the query
        try:
            self.__cur.execute(query)
            self.__conn.commit()

        except psycopg2.Error as e:
            print(e.pgerror)
            raise

        # We reset the cursor
        self._reset_cursor()

    def create_and_fill_table(self,
                              df: pd.DataFrame,
                              table_name: str,
                              types: Dict[str, str],
                              primary_key: Optional[List[str]] = None) -> None:
        """
        Creates a new table and fills it using data from the dataframe "df"

        Args:
            df: pandas dataframe
            table_name: name of the new table
            types: dict with the names of the columns (key) and their respective types (value)
            primary_key: list of column names to use as primary key (or composite key when more than 1)

        Returns: None
        """
        # We first create the table
        self.__create_table(table_name, types, primary_key)

        # We order columns of dataframe according to "types" dictionary
        df = df[types.keys()]

        # We save the df in a temporary csv
        df.to_csv("temp", index=False, na_rep=" ", sep="!")

        # We copy the data from the csv into the table
        file = open("temp", mode="r", newline="\n")
        file.readline()

        # We copy the data to the table
        try:
            self.__cur.copy_from(file, f"{self.__schema}.\"{table_name}\"", sep="!", null=" ")
            self.__conn.commit()
            os.remove("temp")

        except psycopg2.Error as e:
            print(e.pgerror)
            os.remove("temp")
            raise

        # We reset the cursor
        self._reset_cursor()

    def get_table(self,
                  table_name: str,
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieves a table from the database

        Args:
            table_name: name of the table
            columns: list of the columns we want to select (default = None (all columns))

        Returns: pandas dataframe
        """

        # If no column name is specified, we select all columns
        query = "SELECT *" if columns is None else f"SELECT {helpers.colsForSql(columns)} "

        # We add the table name to the query
        query = f"{query} FROM {self.__schema}.\"{table_name}\""

        # We execute the query
        try:
            self.__cur.execute(query)

        except psycopg2.Error as e:
            print(e.pgerror)

        # We retrieve the column names and the data
        columns = [desc[0] for desc in self.__cur.description]
        data = self.__cur.fetchall()

        # We create a pandas dataframe
        df = pd.DataFrame(data=data, columns=columns)

        # We reset the cursor
        self._reset_cursor()

        return df

    def _reset_cursor(self) -> None:
        """
        Resets the cursor
        """
        self.__cur.close()
        self.__cur = self.__conn.cursor()

    def get_all_tables(self) -> List[str]:
        """
        Retrieves the names of all the tables in the specific schema of the database

        Returns: list of table names
        """
        # We execute the query
        try:
            self.__cur.execute(
                f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE"
                f" TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG='{self.__database}'"
                f" AND table_schema = '{self.__schema}' ")

        except psycopg2.Error as e:
            print(e.pgerror)
            raise

        # We extract the tables' names
        tables_names = list(map(lambda t: t[0], self.__cur.fetchall()))

        # We reset the cursor
        self._reset_cursor()

        # We return the names of the tables in the database
        return tables_names

    def get_column_names(self, table_name: str) -> List[str]:
        """
        Retrieves the names of all the columns in a given table

        Args:
            table_name: name of the table

        Returns: list of column names
        """
        table_name = table_name.replace("'", "''")

        try:
            self.__cur.execute(
                f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME= \'{table_name}\'")

        except psycopg2.Error as e:
            print(e.pgerror)
            raise

        # We extract the columns' names
        columns_names = list(map(lambda c: c[0], self.__cur.fetchall()))

        # We reset the cursor
        self._reset_cursor()

        return columns_names

    def get_missing_data_count(self,
                               table_name: str,
                               save_csv: bool = False,
                               excluded_cols: List[str] = ["Remarks"]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Gets the count of all the missing data in the given table

        Args:
            table_name: name of the table
            save_csv: true if we want to save a csv with the results
            excluded_cols: list with names of columns to exclude during the count

        Returns: dataframe with nb of missing data per column, dictionary with details on the missing data
        """

        # We retrieve the table
        df_table = self.get_table(table_name=table_name,
                                  columns=[c for c in self.get_column_names(table_name) if c not in excluded_cols])

        # We count the number of missing values per column
        missing_df = df_table.isnull().sum()

        # We get the counts we need from the dataframe
        missing_count = missing_df.sum()
        complete_row_count = len([complete for complete in missing_df.sum(axis=1) if complete == 0])
        total_rows = df_table.shape[0]

        # We save the csv with the results if required
        if save_csv:
            table_name = helpers.reformat_string(table_name)
            helpers.save_stats_file(table_name, "missing", missing_df, index=True, header=False)

        # returning a dictionary containing the data needed
        return missing_df, {"table_name": table_name,
                            "missing_count": missing_count,
                            "complete_row_count": complete_row_count,
                            "total_rows": total_rows}

    def get_all_missing_data_count(self, filename="petale_missing_data.csv", drawCharts=True):
        """
        Function that generates a csv file containing the count of the missing data of all the tables of the database

        :param filename: the name of file to be generated

        :generates a csv file
        """
        # we initialize the results
        results = []

        # we get all the table names
        tables = self.get_all_tables()

        # For each table we get the missing data count
        with tqdm(total=len(tables)) as pbar:
            for table in tables:
                pbar.update(1)
                _, missingDataCount = self.get_missing_data_count(table, drawCharts)

                # we save the missing data count in results
                results.append(missingDataCount)

        # we generate a csv file from the data in results
        helpers.writeCsvFile(results, filename, "missing_data")
        print("File is ready in the folder missing_data! ")

    def get_common_count(self, tables, columns=[PARTICIPANT, TAG], saveInFile=False):
        """
        Gets the number of common survivors from a list of tables

        :param tables: the list of tables
        :param columns: list of the columns according to we want to get the the common survivors
        :return: number of common survivors

        """
        # we prepare the columns to be in the SQL query
        colsInQuery = helpers.colsForSql(columns)

        # we build the request
        query = 'SELECT COUNT(*) FROM ('
        for index, table in enumerate(tables):
            if index == 0:
                query += f'(SELECT {colsInQuery} FROM \"{table}\")'
            else:
                query += f' INTERSECT (SELECT {colsInQuery} FROM \"{table}\")'
        query += ') l'

        # We execute the query
        try:
            self.cur.execute(query)
        except psycopg2.Error as e:
            print(e.pgerror)

        row = self.cur.fetchall()[0]

        if saveInFile:
            # Saving in file
            if not os.path.isfile("commonPatients.csv"):
                try:
                    with open("commonPatients.csv", 'w', newline='') as csvfile:
                        writer = csv.DictWriter(
                            csvfile, ["tables", "commonSurvivors"])
                        writer.writeheader()
                        separator = " & "
                        writer.writerow({"tables": separator.join(
                            tables), "commonSurvivors": row[0]})
                except IOError:
                    print("I/O error")
            else:
                try:
                    # Open file in append mode
                    with open("commonPatients.csv", 'a+', newline='') as csvfile:
                        # Create a writer object from csv module
                        writer = csv.DictWriter(
                            csvfile, ["tables", "commonSurvivors"])
                        # Add a new row to the csv file
                        separator = " & "
                        writer.writerow({"tables": separator.join(
                            tables), "commonSurvivors": row[0]})
                except IOError:
                    print("I/O error")
        return row[0]

    @staticmethod
    def get_numerical_var_analysis(table_name, df, group=None):
        """

        Function that calculates the mean and variance of variable of a given data frame over all rows,
        and also over groups contained in a specified column "group".

        Male, and Female survivors fot each variable in the given data frame

        :param table_name: name of the table
        :param df: pandas data frame containing the data of numerical variables
        :param group: name of a column, we calculate the stats for the overall data and the stats of the data grouped
        by this column, Ex : group = 34500 Sex will give us the stats for all the data, for Male, and for Female

        return: a pandas data frame
        """
        # we initialize a python dictionary where we will save the results
        results, var_name, all_, group_values = DataManager.__initialize_results_dict(df, group)

        # we get the columns on which we will calculate the stats
        cols = [col for col in df.columns if col != group]

        # for each column we calculate the mean and the variance
        for col in cols:

            # we append the mean and the var for all participants to the results dictionary
            results[var_name].append(col)
            all_mean, all_var, all_min, all_max = helpers.get_column_stats(df, col)
            results[all_].append(f"{all_mean} ({all_var}) [{all_min}, {all_max}]")

            # if the group is given, we calculate the stats for each possible value of that group
            if group is not None:
                for group_val in group_values:
                    # we append the mean and the var for sub group participants to the results dictionary
                    df_group = df[df[group] == group_val]
                    group_mean, group_var, group_min, group_max = helpers.get_column_stats(df_group, col)
                    results[f"{group} {group_val}"].append(f"{group_mean} ({group_var}) [{group_min}, {group_max}]")

        # for each variable of the given dataframe we plot a chart
        folder_name = helpers.reformat_string(table_name)
        for var_name in cols:

            # we plot and save a chart for a single variable
            file_name = helpers.reformat_string(var_name)
            chart_services.drawHistogram(df, var_name, f"Estimated density of {var_name}", file_name, folder_name)

        # we return the results
        return pd.DataFrame(results)

    @staticmethod
    def get_categorical_var_analysis(table_name, df, group=None):
        """Function that calculates the counts and percentage of all the categorical variables given in dataframe
         over all rows, and also over groups contained in a specified column "group".

        :param table_name: name of the table
        :param df: pandas data frame containing the data of numerical variables
        :param group: name of a column, we calculate the stats for the overall data and the stats of the data grouped
        by this column, Ex : group = 34500 Sex will give us the stats for all the data, for Male, and for Female
        return: a pandas dataframe
        """

        # we initialize a python dictionary where we will save the results
        results, var_name, all_, group_values = DataManager.__initialize_results_dict(df, group)

        # we get the columns on which we will calculate the stats
        cols = [col for col in df.columns if col != group]

        # we initialize a python list where we will save data that will be useful when plotting the charts
        data_for_chart = []

        # for each column we calculate the count and the percentage
        for col in cols:

            # we initialize an object that will contain data that will be useful to plot this particular variable
            single_data_for_chart = {"col_name": col, "values": [], "all": []}

            if group is not None:

                group_totals = {}

                for group_val in group_values:
                    single_data_for_chart[group_val] = []
                    group_totals[group_val] = df.loc[df[group] == group_val, col].dropna().shape[0]

            # we get all the categories of this variable
            categories = df[col].dropna().unique()

            # we get the total count
            total = df.shape[0] - df[col].isna().sum()

            # for each category of this variable we get the counts and the percentage
            for category in categories:

                # we get the total count of this category
                category_total = df[df[col] == category].shape[0]

                # we get the total percentage of this category
                all_percent = round(category_total/total * 100, 2)

                # we save the results
                results[var_name].append(f"{col} : {category}")
                results[all_].append(f"{category_total} ({all_percent}%)")

                # we save the data that will be useful when plotting
                single_data_for_chart["values"].append(category)
                single_data_for_chart["all"].append(float(category_total))

                if group is not None:

                    for group_val in group_values:

                        # We create a filter to get the number of items in a group that has the correspond category
                        filter = (df[group] == group_val) & (df[col] == category)

                        # We compute the statistics needed
                        sub_category_total = df[filter].shape[0]
                        sub_category_percent = round(sub_category_total/(group_totals[group_val]) * 100, 2)
                        results[f"{group} {group_val}"].append(f"{sub_category_total} ({sub_category_percent}%)")

                        # We save data for the charts
                        single_data_for_chart[group_val].append(float(sub_category_total))

            data_for_chart.append(single_data_for_chart)

        # we make a chart from this analysis
        for item in data_for_chart:
            if group is not None:

                # plotting the chart
                filename = helpers.reformat_string(item["col_name"])
                folder_name = helpers.reformat_string(table_name)
                chart_services.drawBinaryGroupedBarChart(
                    item["values"], {"label": "Men", "values": item["Men"]},
                    {"label": "Women", "values": item["Women"]}, "Categories", "Count", item["col_name"],
                    f"chart_{filename}", f"charts_{folder_name}")

        # we return the data frame containing the informations
        return pd.DataFrame(results)

    @staticmethod
    def get_group_count(df, group):
        """
        Count the number of items from each group

        :param df: pandas dataframe
        :param group: name of a column, we calculate the stats for the overall data and the stats of the data grouped
        by this column, Ex : group = 34500 Sex will give us the stats for all the data, for Male, and for Female
        :return: pandas dataframe
        """

        # we initialize a python dictionary where we will save the results
        results, var_name, all_, group_values = DataManager.__initialize_results_dict(df, group)

        # We set the variable name as "n"
        results[var_name].append("n")

        # We count the total number of rows
        results[all_].append(df.shape[0])

        # We count the number of rows from each group
        for group_val in group_values:
            results[f"{group} {group_val}"].append(df[df[group] == group_val].shape[0])

        # We return the resulting dataframe
        return pd.DataFrame(results)

    @staticmethod
    def __initialize_results_dict(df, group):

        var_name = "Variable Name"
        all_ = "All"
        group_values = None

        if group is None:
            results = {
                var_name: [],
                all_: []
            }
        else:
            group_values = df[group].unique()
            results = {
                var_name: [],
                all_: []
            }
            for group_val in group_values:
                results[f"{group} {group_val}"] = []

        return results, var_name, all_, group_values


class PetaleDataManager(DataManager):

    def __init__(self, user, host='localhost', port='5437'):
        super().__init__(user, 'petale101', 'petale', host, port, 'public')

    def get_table_stats(self, table_name, include="ALL",
                        exclude=["Date", "Form", "Status", "Remarks"], save_in_file=True):
        """
        Function that return a dataframe containing statistics from any given table of the PETALE database

        :param table_name: The name of the table
        :param include: a list of all the columns to include, "ALL" if all columns included
        :param exclude: a list of all the columns to exclude
        :param save_in_file: Boolean, if true the dataframe will be saved in a csv file
        :return: pandas DataFrame
        """
        # we get a dataframe for the given tablename
        if include == "ALL":
            table_df = self.get_table(table_name)
        else:
            table_df = self.get_table(table_name, include)

        # we retrieve the columns of the table
        cols = table_df.columns

        # we exclude the parameters specified with the exclude parameter
        cols = [col for col in cols if col not in exclude]
        table_df = table_df[cols]

        # we get only the rows that satisfy the given conditions
        if TAG in cols:
            table_df = table_df[table_df[TAG] == PHASE]
            table_df = table_df.drop([TAG], axis=1)

        # We get the dataframe from the table the table containing the sex information
        if SEX in cols:
            sex_df = table_df[[PARTICIPANT, SEX]]
            table_df = table_df.drop([SEX], axis=1)
        else:
            sex_df = self.get_table(GEN_1, columns=[PARTICIPANT, TAG, SEX])
            sex_df = sex_df[sex_df[TAG] == PHASE]
            sex_df = sex_df.drop([TAG], axis=1)

        # we retrieve categorical and numerical data
        categorical_df = helpers.retrieve_categorical(table_df, ids=[PARTICIPANT])
        numerical_df = helpers.retrieve_numerical(table_df, ids=[PARTICIPANT])

        # we merge the the categorical dataframe with the general dataframe by the column PARTICIPANT
        categorical_df = pd.merge(sex_df, categorical_df, on=PARTICIPANT, how=INNER)
        categorical_df = categorical_df.drop([PARTICIPANT], axis=1)

        # we merge the the numerical dataframe with the general dataframe by the column PARTICIPANT
        numerical_df = pd.merge(sex_df, numerical_df, on=PARTICIPANT, how=INNER)
        numerical_df = numerical_df.drop([PARTICIPANT], axis=1)

        # we retrieve number of individuals from each sex
        sex_stats = self.get_group_count(numerical_df, group=SEX)

        # we make a categorical var analysis for this table
        categorical_stats = self.get_categorical_var_analysis(table_name, categorical_df, group=SEX)

        # we make a numerical var analysis for this table
        numerical_stats = self.get_numerical_var_analysis(table_name, numerical_df, group=SEX)

        # we concatenate all the results to get the final stats dataframe
        stats_df = pd.concat([sex_stats, categorical_stats, numerical_stats], ignore_index=True)
        table_name = helpers.reformat_string(table_name)

        # if saveInFile True we save the dataframe in a csv file
        if save_in_file:
            helpers.save_stats_file(table_name, "statistics", stats_df)

        # we return the dataframe
        return stats_df

    def get_variable_info(self, var_name):
        """
        Function that returns all the information about a specific variable

        :param var_name: The name of the variable
        :return: a python dictionary containing all the infos
        """
        # we extract the variable id from the variable name
        var_id = helpers.extract_var_id(var_name)

        # we prepare the query
        query = f'SELECT * FROM "PETALE_meta_data" WHERE "Test ID" = {var_id}'

        # We execute the query
        try:
            self.cur.execute(query)
        except psycopg2.Error as e:
            print(e.pgerror)

        row = self.cur.fetchall()

        # we initialize the dictionary that will contain the result
        var_info = {}

        # we create an array containing all the keys
        var_info_labels = ["test_id", "editorial_board", "form", "number",
                           "section", "test", "description", "type", "option", "unit", "remarks"]

        # we fill the dictionary with informations
        for index, data in enumerate(row[0]):
            var_info[var_info_labels[index]] = data

        # We reset the cursor
        self.reset_cursor()

        # we return the result
        return var_info

    def get_id_conversion_map(self):
        """
        Returns a map that links genomic patients' reference names to their IDs
        :return: dict
        """
        conversion_df = self.get_table(PETALE_PANDORA)
        conversion_map = {k: v[0] for k, v in conversion_df.set_index('Reference name').T.to_dict('series').items()}

        return conversion_map


def initialize_petale_data_manager():
    """
    Asks petale database user name to initialise a PetaleDataManager object
    :return: PetaleDataManager
    """
    user_name = input("Enter your username to access PETALE database : ")
    return PetaleDataManager(user_name)

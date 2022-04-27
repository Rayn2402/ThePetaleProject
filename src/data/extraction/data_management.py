"""
Filename: data_management.py

Author: Nicolas Raymond
        Mehdi Mitiche

Description: Defines the DataManager class that helps interaction with SQL data

Date of last modification : 2021/11/04
"""

import csv
import psycopg2
import pandas as pd
import os

from settings.database import *
from settings.paths import Paths
from src.data.extraction import helpers
from src.data.extraction.constants import *
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple


class DataManager:
    """
    Object that can interact with a PostgresSQL database
    """
    # Common count csv headers constants
    TABLES: str = "tables"
    COMMON_ELEMENTS: str = "common elements"

    # Results dictionary keys constants
    VAR_NAME: str = "Variable Name"
    ALL: str = "All"

    # Temporary table name constant
    TEMP: str = "temp"

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
        self.__conn = DataManager._connect(user, password, database, host, port)
        self.__schema = schema
        self.__database = database

    def _create_table(self,
                      table_name: str,
                      types: Dict[str, str],
                      primary_key: Optional[List[str]] = None) -> None:
        """
        Creates a table named "table_name" that has the columns with the types indicated
        in the dictionary "types".

        Args:
            table_name: name of the table
            types: dictionary with names of the columns (key) and their respective types (value)
            primary_key: list of column names to use as primary key (or composite key when more than 1)

        Returns:
        """
        # We save the start of the query
        query = f"CREATE TABLE {self.__schema}.\"{table_name}\" (" + self._reformat_columns_and_types(types)

        # If a primary key is given we add it to the query
        if primary_key is not None:

            # We define the primary key
            keys = self._reformat_columns(primary_key)
            query += f", PRIMARY KEY ({keys}) );"

        else:
            query += ");"

        # We execute the query
        with self.__conn.cursor() as curs:
            try:
                curs.execute(query)
                self.__conn.commit()

            except psycopg2.Error as e:
                raise Exception(e.pgerror)

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
        self._create_table(table_name, types, primary_key)

        # We order columns of dataframe according to "types" dictionary
        df = df[types.keys()]

        # We save the df in a temporary csv
        df.to_csv(DataManager.TEMP, index=False, na_rep=" ", sep="!")

        # We copy the data from the csv into the table
        file = open(DataManager.TEMP, mode="r", newline="\n")
        file.readline()

        # We copy the data to the table
        with self.__conn.cursor() as curs:
            try:
                curs.copy_from(file, f"{self.__schema}.\"{table_name}\"", sep="!", null=" ")
                self.__conn.commit()
                os.remove(DataManager.TEMP)

            except psycopg2.Error as e:
                os.remove(DataManager.TEMP)
                raise Exception(e.pgerror)

    def get_all_missing_data_count(self,
                                   directory: Optional[str] = None,
                                   filename: str = "missing_data.csv") -> None:
        """
        Generates a csv file containing the count of the missing data of all the tables in the database

        Args:
            directory: path of the directory where the results will be saved
            filename: name of the file containing the results

        Returns: None
        """
        # We get all the table names
        tables = self.get_all_table_names()

        # For each table we get the missing data count
        with tqdm(total=len(tables)) as pbar:
            results = []
            for table in tables:

                # We extract the count of missing data
                _, missing_data_count = self.get_missing_data_count(table)

                # We save the missing data count in results
                results.append(missing_data_count)

                # We update the progress bar
                pbar.update(1)

        # We generate a csv file with all the results
        directory = directory if directory is not None else os.getcwd()
        self.write_csv_from_dict(data=results, filename=filename, directory=directory)
        print(f"File is ready in the folder {directory}")

    def get_all_table_names(self) -> List[str]:
        """
        Retrieves the names of all the tables in the specific schema of the database

        Returns: list of table names
        """
        # We execute the query
        with self.__conn.cursor() as curs:
            try:
                curs.execute(
                    f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE"
                    f" TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG='{self.__database}'"
                    f" AND table_schema = '{self.__schema}' ")

                # We extract the tables' names
                tables_names = list(map(lambda t: t[0], curs.fetchall()))

            except psycopg2.Error as e:
                raise Exception(e.pgerror)

        # We return the names of the tables in the database
        return tables_names

    def get_categorical_var_analysis(self,
                                     df: pd.DataFrame,
                                     group: Optional[str] = None) -> pd.DataFrame:
        """
        Calculates the counts and percentage of all the categorical variables given in dataframe
        over all rows, and also over groups contained in a specified column "group"

        Args:
            df: pandas dataframe with the categorical variables
            group: name of a column, Ex : group = "Sex" will give us the stats for all the data, for men, and for women

        Returns: pandas dataframe with the statistics
        """

        # We initialize a python dictionary in which we will save the results
        results, group_values = self._initialize_results_dict(df, group)

        # We get the columns on which we will calculate the stats
        cols = [col for col in df.columns if col != group]

        # We initialize a dictionary that will store the totals of each group within the column "group"
        group_totals = {}

        # For each column we calculate the count and the percentage
        for col in cols:

            # We get all the categories of this variable
            categories = df[col].dropna().unique()

            # We get the total count
            total = df.shape[0] - df[col].isna().sum()

            # For each category of this variable we get the counts and the percentage
            for category in categories:

                # We get the total count of this category
                category_total = df[df[col] == category].shape[0]

                # We get the total percentage of this category
                all_percent = round(category_total/total * 100, 2)

                # We save the results
                results[DataManager.VAR_NAME].append(f"{col} : {category}")
                results[DataManager.ALL].append(f"{category_total} ({all_percent}%)")

                if group is not None:
                    for group_val in group_values:

                        # We get the number of elements within the group
                        group_totals[group_val] = df.loc[df[group] == group_val, col].dropna().shape[0]

                        # We create a filter to get the number of items in a group that has the correspond category
                        filter_ = (df[group] == group_val) & (df[col] == category)

                        # We compute the statistics needed
                        sub_category_total = df[filter_].shape[0]
                        sub_category_percent = round(sub_category_total/(group_totals[group_val]) * 100, 2)
                        results[f"{group} {group_val}"].append(f"{sub_category_total} ({sub_category_percent}%)")

        return pd.DataFrame(results)

    def get_column_names(self, table_name: str) -> List[str]:
        """
        Retrieves the names of all the columns in a given table

        Args:
            table_name: name of the table

        Returns: list of column names
        """
        table_name = table_name.replace("'", "''")
        with self.__conn.cursor() as curs:
            try:
                curs.execute(
                    f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME= \'{table_name}\'")

                # We extract the columns' names
                columns_names = list(map(lambda c: c[0], curs.fetchall()))
            except psycopg2.Error as e:
                raise Exception(e.pgerror)

        return columns_names

    def get_common_count(self,
                         tables: List[str],
                         columns: List[str],
                         filename: Optional[str] = None) -> int:
        """
        Retrieves the number of common elements in multiple tables, using the values
        of the given columns to identify the unique elements in each table

        Args:
            tables: list of table names
            columns: list of columns identifying unique elements (composite key)
            filename: if provided, will be the name of the csv file containing the results

        Returns: number of common elements
        """
        # We prepare the columns to be in the SQL query
        cols_in_query = self._reformat_columns(columns)

        # We build the query
        query = 'SELECT COUNT(*) FROM ('
        for index, table in enumerate(tables):
            if index == 0:
                query += f'(SELECT {cols_in_query} FROM \"{table}\")'
            else:
                query += f' INTERSECT (SELECT {cols_in_query} FROM \"{table}\")'
        query += ') l'

        with self.__conn.cursor() as curs:
            try:

                # We execute the query
                curs.execute(query)

                # We extract the common count
                count = curs.fetchall()[0][0]

            except psycopg2.Error as e:
                print(e.pgerror)

        if filename is not None:

            # If the file does not exist
            if not os.path.isfile(filename):

                # We will need to write a header and use the opening mode "write"
                write_header, mode = True, "w"

            else:
                # We will use the opening mode "append"
                write_header, mode = False, "a+"

            # We try to write the results
            try:
                with open(f"{filename}.csv", mode, newline='') as csv_file:
                    writer = csv.DictWriter(csv_file, [DataManager.TABLES, DataManager.COMMON_ELEMENTS])
                    if write_header:
                        writer.writeheader()
                    writer.writerow({DataManager.TABLES: " & ".join(tables), DataManager.COMMON_ELEMENTS: count})
            except IOError:
                raise Exception("I/O error")

        return count

    def get_group_count(self,
                        df: pd.DataFrame,
                        group: str) -> pd.DataFrame:
        """
        Count the number of items within each group of the categorical variable "group"

        Args:
            df: pandas dataframe
            group: name of a categorical column

        Returns: pandas dataframe with the counts
        """

        # We initialize a python dictionary where we will save the results
        results, group_values = self._initialize_results_dict(df, group)

        # We set the variable name as "n"
        results[DataManager.VAR_NAME].append("n")

        # We count the total number of rows
        results[DataManager.ALL].append(df.shape[0])

        # We count the number of rows from each group
        for group_val in group_values:
            results[f"{group} {group_val}"].append(df[df[group] == group_val].shape[0])

        return pd.DataFrame(results)

    def get_missing_data_count(self,
                               table_name: str,
                               directory: Optional[str] = None,
                               excluded_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Gets the count of all the missing data in the given table

        Args:
            table_name: name of the table
            directory: if provided, a csv with the results will be saved at the given directory
            excluded_cols: list with names of columns to exclude during the count

        Returns: dataframe with nb of missing data per column, dictionary with details on the missing data
        """

        # We retrieve the table
        excluded_cols = excluded_cols if excluded_cols is not None else []
        df_table = self.get_table(table_name=table_name,
                                  columns=[c for c in self.get_column_names(table_name) if c not in excluded_cols])

        # We count the number of missing values per column
        missing_df = df_table.isnull()

        # We get the counts we need from the dataframe
        missing_count_by_col = missing_df.sum()
        complete_row_count = len([complete for complete in missing_df.sum(axis=1) if complete == 0])

        # We save the csv with the results if required
        if directory is not None:
            table_name = self._reformat_table_name(table_name)
            file_path = os.path.join(directory, f"{table_name}_missing_count.csv")
            missing_df.to_csv(file_path, index=True, header=False)

        # Returning a dictionary containing the data needed
        return missing_count_by_col, {"table_name": table_name,
                                      "missing_count": missing_df.sum().sum(),
                                      "complete_row_count": complete_row_count,
                                      "nb_rows": df_table.shape[0],
                                      "nb_columns": df_table.shape[1]}

    def get_numerical_var_analysis(self,
                                   df: pd.DataFrame,
                                   group: Optional[str] = None) -> pd.DataFrame:
        """
        Calculates the mean, the variance, the min and the max of variables
        within a given data frame over all rows, and also over groups contained in a specified column "group".

        Args:
            df: pandas data frame containing the data of numerical variables
            group: name of a column, Ex : group = "Sex" will give us the stats for all the data, for men, and for women

        Returns: pandas dataframe with the statistics
        """
        # We initialize a python dictionary in which we will save the results
        results, group_values = self._initialize_results_dict(df, group)

        # We get the columns on which we will calculate the stats
        cols = [col for col in df.columns if col != group]

        # For each column we calculate the mean and the variance
        for col in cols:

            # We append the statistics for all participants to the results dictionary
            results[DataManager.VAR_NAME].append(col)
            all_mean, all_var, all_min, all_max = helpers.get_column_stats(df, col)
            results[DataManager.ALL].append(f"{all_mean} ({all_var}) [{all_min}, {all_max}]")

            # If a group column is given, we calculate the stats for each possible value of that group
            if group is not None:
                for group_val in group_values:

                    # We append the statistics for sub group participants to the results dictionary
                    df_group = df[df[group] == group_val]
                    group_mean, group_var, group_min, group_max = helpers.get_column_stats(df_group, col)
                    results[f"{group} {group_val}"].append(f"{group_mean} ({group_var}) [{group_min}, {group_max}]")

        return pd.DataFrame(results)

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
        query = "SELECT *" if columns is None else f"SELECT {self._reformat_columns(columns)} "

        # We add the table name to the query
        query = f"{query} FROM {self.__schema}.\"{table_name}\""

        with self.__conn.cursor() as curs:
            try:
                # We execute the query
                curs.execute(query)

                # We retrieve the column names and the data
                columns = [desc[0] for desc in curs.description]
                data = curs.fetchall()

            except psycopg2.Error as e:
                raise Exception(e.pgerror)

        # We create a pandas dataframe
        df = pd.DataFrame(data=data, columns=columns)

        return df

    @staticmethod
    def _connect(user: str,
                 password: str,
                 database: str,
                 host: str,
                 port: str) -> Any:
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

        except psycopg2.Error as e:
            raise Exception(e.pgerror)

        return conn

    @staticmethod
    def _initialize_results_dict(df: pd.DataFrame,
                                 group: str) -> Tuple[Dict[str, List[Any]], List[Any]]:
        """
        Initializes a dictionary that will contains results of a descriptive analyses

        Args:
            df: pandas dataframe
            group: name of a column

        Returns: initialized dictionary, list with the different group values
        """
        group_values = None
        results = {
            DataManager.VAR_NAME: [],
            DataManager.ALL: []
        }
        if group is not None:
            group_values = df[group].unique()
            for group_val in group_values:
                results[f"{group} {group_val}"] = []

        return results, group_values

    @staticmethod
    def _reformat_columns(cols: List[str]) -> str:
        """
        Converts a list of strings containing the name of some columns to a string
        ready to use in an SQL query. Ex: ["name","age","gender"]  ==> "name","age","gender"

        Args:
            cols: list of column names

        Returns: str to include in a query
        """
        cols = list(map(lambda c: '"'+c+'"', cols))

        return ",".join(cols)

    @staticmethod
    def _reformat_columns_and_types(types: Dict[str, str]) -> str:
        """
        Converts a dictionary with column names as keys and types as values to a string to
        use in an SQL query. Ex: {"name":"text", "Age":"numeric"} ==> '"name" text, "Age" numeric'

        Args:
            types: dictionary with column names (keys) and types (values)

        Returns: str to include in a query
        """
        cols = types.keys()
        query_parts = list(map(lambda c: f"\"{c}\" {types[c]}", cols))

        return ",".join(query_parts)

    @staticmethod
    def _reformat_table_name(table_name: str) -> str:
        """
        Removes special characters in a table name so it can be used as a directory name

        Args:
            table_name: str

        Returns:
        """
        return table_name.replace(".", "").replace(": ", "").replace("?", "").replace("/", "")

    @staticmethod
    def write_csv_from_dict(data: List[Dict[str, Any]],
                            filename: str,
                            directory: str) -> None:
        """
        Takes a list of dictionaries sharing the same keys and generates a CSV file from them

        Args:
            data: list of dictionaries
            filename: name of the csv file that will be generated
            directory: directory where the file will be saved

        Returns: None
        """
        try:
            # We create the directory if it does not exist
            os.makedirs(directory, exist_ok=True)

            # We write in the new csv file
            with open(os.path.join(directory, filename), 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
                writer.writeheader()
                for item in data:
                    writer.writerow(item)

        except IOError:
            raise Exception("I/O error")


class PetaleDataManager(DataManager):
    """
    DataManager that has specific methods for Petale database
    """
    # Constant related to variable meta data file
    VARIABLE_INFOS = ["test_id", "editorial_board", "form", "number",
                      "section", "test", "description", "type", "option", "unit", "remarks"]

    # Constant related to snp dataframe column
    KEY = "CHROM_POS"

    def __init__(self):
        """
        Sets the private attributes using database configurations in settings/database.py
        """
        super().__init__(user=USER,
                         password=PASSWORD,
                         database=DATABASE,
                         host=HOST,
                         port=PORT,
                         schema=SCHEMA)

    def get_common_survivor(self,
                            tables: List[str],
                            filename: Optional[str] = None) -> int:
        """
        Retrieves the number of common survivors among multiple tables

        Args:
            tables: list of table names
            filename: if provided, will be the name of the csv file containing the results

        Returns: number of common survivors
        """
        return self.get_common_count(tables=tables, columns=[PARTICIPANT, TAG], filename=filename)

    def get_missing_data_count(self,
                               table_name: str,
                               directory: Optional[str] = Paths.DESC_STATS,
                               excluded_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Gets the count of all the missing data in the given table.
        Remarks column will be automatically excluded from the count.

        Args:
            table_name: name of the table
            directory: if provided, a csv with the results will be saved at the given directory
            excluded_cols: list with names of columns to exclude during the count

        Returns: dataframe with nb of missing data per column, dictionary with details on the missing data
        """
        # We add Remarks to excluded columns
        if excluded_cols is None:
            excluded_cols = []
        excluded_cols.append(REMARKS)

        return DataManager.get_missing_data_count(self, table_name, directory, excluded_cols)

    def get_table_stats(self,
                        table_name: str,
                        included_cols: Optional[List[str]] = None,
                        excluded_cols: Optional[List[str]] = None,
                        save_in_file: bool = True) -> pd.DataFrame:
        """
        Retrieves numerical and categorical statistics of a Petale table

        Args:
            table_name: name of the table
            included_cols: list of all the columns to include (default = None (all columns))
            excluded_cols: list of all the columns to exclude (default = None)
            save_in_file: true if we want to save the statistics in a csv file associated to the table name

        Returns: dataframe with all the statistics
        """

        # We get a the dataframe containing the table
        if included_cols is None:
            table_df = self.get_table(table_name)
        else:
            table_df = self.get_table(table_name, included_cols)

        # We update the list of all columns that must be excluded
        if excluded_cols is None:
            excluded_cols = []
        excluded_cols += [DATE, FORM, STATUS, REMARKS]

        # We exclude the variables specified
        cols = [col for col in table_df.columns if col not in excluded_cols]
        table_df = table_df[cols]

        # We get only the rows associated to a specific phase
        if TAG in cols:
            table_df = table_df[table_df[TAG] == PHASE]
            table_df = table_df.drop([TAG], axis=1)

        # We get the dataframe from the table containing the sex information
        if SEX in cols:
            sex_df = table_df[[PARTICIPANT, SEX]]
            table_df = table_df.drop([SEX], axis=1)
        else:
            sex_df = self.get_table(GEN_1, columns=[PARTICIPANT, TAG, SEX])
            sex_df = sex_df[sex_df[TAG] == PHASE]
            sex_df = sex_df.drop([TAG], axis=1)

        # We retrieve categorical and numerical data
        categorical_df = helpers.retrieve_categorical_var(table_df, to_keep=[PARTICIPANT])
        numerical_df = helpers.retrieve_numerical_var(table_df, to_keep=[PARTICIPANT])

        # We merge the the categorical dataframe with the sex dataframe by the column PARTICIPANT
        categorical_df = pd.merge(sex_df, categorical_df, on=PARTICIPANT, how=INNER)
        categorical_df = categorical_df.drop([PARTICIPANT], axis=1)

        # We merge the the numerical dataframe with the sex dataframe by the column PARTICIPANT
        numerical_df = pd.merge(sex_df, numerical_df, on=PARTICIPANT, how=INNER)
        numerical_df = numerical_df.drop([PARTICIPANT], axis=1)

        # We retrieve number of individuals from each sex
        sex_stats = self.get_group_count(numerical_df, group=SEX)

        # We make a categorical variable analysis for this table
        categorical_stats = self.get_categorical_var_analysis(categorical_df, group=SEX)

        # we make a numerical variable analysis for this table
        numerical_stats = self.get_numerical_var_analysis(numerical_df, group=SEX)

        # We concatenate all the results to get the final stats dataframe
        stats_df = pd.concat([sex_stats, categorical_stats, numerical_stats], ignore_index=True)
        table_name = self._reformat_table_name(table_name)

        if save_in_file:
            self._save_stats_file(table_name, "statistics", stats_df)

        return stats_df

    def get_variable_info(self, var_name: str) -> Dict[str, Any]:
        """
        Retrieves all the information about a specific variable from the original tables

        Args:
            var_name: name of the variable

        Returns: dictionary with all the infos
        """
        # We extract the variable id from the variable name
        var_id = self.extract_var_id(var_name)

        # we prepare the query
        query = f'SELECT * FROM "PETALE_meta_data" WHERE "Test ID" = {var_id}'

        # We execute the query
        try:
            self.__cur.execute(query)
        except psycopg2.Error as e:
            print(e.pgerror)

        # We create the dictionary with the results
        var_info = {PetaleDataManager.VARIABLE_INFOS[i]: data for i, data in enumerate(self.__cur.fetchall()[0])}

        # We reset the cursor
        self._reset_cursor()

        return var_info

    def get_id_conversion_map(self) -> Dict[str, str]:
        """
        Returns a map that links patients' genomic reference names to their IDs

        Returns: dictionary
        """
        conversion_df = self.get_table(PETALE_PANDORA)
        conversion_map = {k: v[0] for k, v in conversion_df.set_index('Reference name').T.to_dict('series').items()}

        return conversion_map

    @staticmethod
    def _save_stats_file(table_name: str,
                         file_name: str,
                         df: pd.DataFrame,
                         index: bool = False,
                         header: bool = True) -> None:
        """
        Saves a csv file in the stats directory associated to a table

        Args:
            table_name: name of the table for which we save the statistics
            file_name: name of the csv file
            df: pandas dataframe to turn into csv
            index: true if we need to add indexes in the csv
            header: true if we need to store the header of the df in the csv

        Returns: None
        """
        # We save the file path
        directory = os.path.join(Paths.DESC_STATS, table_name)
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{file_name}.csv")

        # We remove the current csv if it is already existing
        if os.path.isfile(file_path):
            os.remove(file_path)

        df.to_csv(file_path, index=index, header=header)

    @staticmethod
    def extract_var_id(var_name: str) -> str:
        """
        Extracts the id of the given variable

        Args:
            var_name: variable name from a Petale original table

        Returns: id of the variable
        """
        return var_name.split()[0]

    @staticmethod
    def fill_participant_id(participant_id: str) -> str:
        """
        Adds characters missing to a participant ID

        Args:
            participant_id: current ID

        Returns: corrected participant ID
        """

        return f"P" + "".join(["0"]*(3-len(participant_id))) + participant_id

    @staticmethod
    def pivot_snp_dataframe(df: pd.DataFrame,
                            snps_id_filter: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Filters the patients snps table and execute a transposition

        Args:
            df: pandas dataframe
            snps_id_filter: list with snps id to keep

        Returns: pandas dataframe
        """

        # We add a new column to the dataframe
        df[PetaleDataManager.KEY] = df[CHROM].astype(str) + "_" + df[SNPS_POSITION].astype(str)

        # We filter the table to only keep rows where CHROM and POS match with an SNP in the filter
        if snps_id_filter is not None:
            df = df[df[PetaleDataManager.KEY].isin(list(snps_id_filter[PetaleDataManager.KEY].values))]

        # We dump CHROM and POS columns
        df = df.drop([CHROM, SNPS_POSITION, REF, ALT, GENE_REF_GEN], axis=1, errors='ignore')

        # We change index for CHROM_POS column
        df = df.set_index(PetaleDataManager.KEY)

        # We transpose the dataframe
        df = df.T
        df.index.rename(PARTICIPANT, inplace=True)
        df.reset_index(inplace=True)

        return df

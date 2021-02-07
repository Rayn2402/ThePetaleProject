"""

Authors : Nicolas Raymond
          Mehdi Mitiche

This file contains all functions linked to SQL data management

"""
from . import ChartServices as ChartServices
from . import Helpers as Helpers
import psycopg2
import pandas as pd
import os
import csv
from tqdm import tqdm


class DataManager:

    def __init__(self, user, password, database, host='localhost', port='5432', schema='public'):
        """
        Object that can interact with a PostgreSQL database

        :param user: username to access 'database'
        :param password: password linked to the user
        :param database: name of the database
        :param host: database host address
        :param port: connection port number
        """
        self.conn, self.cur = DataManager.connect(
            user, password, database, host, port)
        self.schema = schema
        self.database = database

    @staticmethod
    def connect(user, password, database, host, port):
        """
        Creates a connection to the database

        :param user: username to access 'database'
        :param password: password linked to the user
        :param database: name of the database
        :param host: database host address
        :param port: connection port number
        :return: connection and cursor
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

    def create_table(self, table_name, types, primary_key=None):
        """
        Creates a table named "table_name" that as columns and types indicates in the dict "types".

        :param table_name: name of the table
        :param types: names of the columns (key) and their respective types (value) in a dict
        :param primary_key: list of column names to use as primary key (or composite key when more than 1)
        """

        query = f"CREATE TABLE {self.schema}.\"{table_name}\" (" + Helpers.colsAndTypes(
            types)

        if primary_key is not None:

            # We define the primary key
            keys = Helpers.colsForSql(primary_key)
            query += f", PRIMARY KEY ({keys}) );"
        else:
            query += ");"

        # We execute the query
        try:
            self.cur.execute(query)
            self.conn.commit()

        except psycopg2.Error as e:
            print(e.pgerror)
            raise

        # We reset the cursor
        self.reset_cursor()

    def create_and_fill_table(self, df, table_name, types, primary_key=None):
        """
        Creates a new table and fill it using data from the dataframe "df"

        :param df: pandas dataframe
        :param table_name: name of the new table
        :param types: names of the columns (key) and their respective types (value) in a dict
        :param primary_key: list of column names to use as primary key (or composite key when more than 1)
        """
        # We first create the table
        self.create_table(table_name, types, primary_key)

        # We save the df in a temporary csv
        df.to_csv("temp", index=False, na_rep=" ", sep="!")

        # We copy the data from the csv into the table
        file = open("temp", mode="r", newline="\n")
        file.readline()

        # We copy the data to the table
        try:
            self.cur.copy_from(
                file, f"{self.schema}.\"{table_name}\"", sep="!", null=" ")
            self.conn.commit()
            os.remove("temp")

        except psycopg2.Error as e:
            print(e.pgerror)
            os.remove("temp")
            raise

        # We reset the cursor
        self.reset_cursor()

    def get_table(self, table_name, columns=None):
        """
        Retrieves a table from the database

        :param table_name: name of the table
        :param columns: list of the columns we want to select (default : None (all columns))
        :return: Pandas dataframe

        """
        query = "SELECT"

        # If no column name is specified, we select all columns
        if columns is None:
            query = f"{query} *"

        # Else, we add the corresponding column names to the query
        else:
            query = query
            columnsToFetch = Helpers.colsForSql(columns)
            query = f"{query} {columnsToFetch}"

        # We add table name to the query
        query = f"{query} FROM {self.schema}.\"{table_name}\""

        # We execute the query
        try:
            self.cur.execute(query)

        except psycopg2.Error as e:
            print(e.pgerror)

        # We retrieve column names and data
        columns = [desc[0] for desc in self.cur.description]
        data = self.cur.fetchall()

        # We create a pandas dataframe
        df = pd.DataFrame(data=data, columns=columns)

        # We reset the cursor
        self.reset_cursor()

        return df

    def reset_cursor(self):
        """
        Resets the cursor
        """
        self.cur.close()
        self.cur = self.conn.cursor()

    def get_all_tables(self):
        """
        Retrieves the names of all the tables of the database

        :return: list of strings

        """
        # We execute the query
        try:
            self.cur.execute(
                f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE"
                f" TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG='{self.database}' AND table_schema = '{self.schema}' ")
        except psycopg2.Error as e:
            print(e.pgerror)
            raise

        tables = self.cur.fetchall()

        # We reset the cursor
        self.reset_cursor()

        # We return the names of the tables in the database
        return list(map(lambda t: t[0], tables))

    def get_column_names(self, tableName):
        """
        Function that retrieves the names of all the columns in a given table

        :param tableName : the name of the table
        :return: list of strings

        """
        escapedTableName = tableName.replace("'", "''")

        try:
            self.cur.execute(
                f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME= \'{escapedTableName}\'")
        except psycopg2.Error as e:
            print(e.pgerror)
            raise

        cols = self.cur.fetchall()
        cols = list(map(lambda c: c[0], cols))

        # We reset the cursor
        self.reset_cursor()
        return cols

    def get_missing_data_count(self, tableName, drawChart=False, excludedCols=["Remarks"]):
        """
        get the count of all the missing data of one given table

        :param tableName: name of the table
        :param drawChart: boolean to indicate if a chart should be created to visualize the missing data
        :param: excludedCols: list of strings containing the list of columns to exclude
        :return: a python dictionary containing the missing data count and the number of complete rows

        """

        # Extracting the name of the columns of the given  table
        cols = self.get_column_names(tableName)

        # Excluding the non needed columns
        cols = [col for col in cols if col not in excludedCols]

        # We retrieve the table
        df_table = self.get_table(tableName, cols)

        missing_x = df_table.isnull().sum().values

        # we get the counts we need from the dataframe
        missingCount = df_table.isnull().sum().sum()
        completedRowCount = len(
            [complete for complete in df_table.isnull().sum(axis=1) if complete == 0])
        totalRows = df_table.shape[0]

        # Plotting the bar chart
        if drawChart:
            fileName = "missing_data_" + \
                tableName.replace(".", "").replace(":", "").replace("/", "")
            folderName = "missing_data_charts"
            figureTitle = f'Count of missing data by columns names for the table {tableName}'
            ChartServices.drawBarhChart(
                cols, missing_x, "Columns", "Data missing", figureTitle, fileName, folderName)

        # returning a dictionary containing the data needed
        return {"tableName": tableName, "missingCount": missingCount,
                "completedRowCount": completedRowCount, "totalRows": totalRows}

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
                missingDataCount = self.get_missing_data_count(
                    table, drawCharts)

                # we save the missing data count in results
                results.append(missingDataCount)

        # we generate a csv file from the data in results
        Helpers.writeCsvFile(results, filename, "missing_data")
        print("File is ready in the folder missing_data! ")

    def get_common_count(self, tables, columns=["Participant", "Tag"], saveInFile=False):
        """
        Gets the number of common survivors from a list of tables

        :param tables: the list of tables
        :param columns: list of the columns according to we want to get the the common survivors
        :return: number of common survivors

        """
        # we prepare the columns to be in the SQL query
        colsInQuery = Helpers.colsForSql(columns)

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
        results, var_name, all_, group_values = DataManager.__initialize_results_dict(
            df, group)

        # we get the columns on which we will calculate the stats
        cols = [col for col in df.columns if col != group]

        # for each column we calculate the mean and the variance
        for col in cols:

            # we append the mean and the var for all participants to the results dictionary
            results[var_name].append(col)
            all_mean = round(df[col].astype("float").mean(axis=0), 2)
            all_var = round(df[col].astype("float").var(axis=0), 2)
            all_max = df[col].astype("float").max()
            all_min = df[col].astype("float").min()
            results[all_].append(
                f"{all_mean} ({all_var}) [{all_min}, {all_max}]")

            # if the group is given, we calculate the stats for each possible value of that group
            if group is not None:
                for group_val in group_values:
                    # we append the mean and the var for sub group participants to the results dictionary
                    df_group = df[df[group] == group_val]
                    group_mean = round(
                        df_group[col].astype("float").mean(axis=0), 2)
                    group_var = round(
                        df_group[col].astype("float").var(axis=0), 2)
                    group_max = df_group[col].astype("float").max()
                    group_min = df_group[col].astype("float").min()
                    results[f"{group} {group_val}"].append(
                        f"{group_mean} ({group_var}) [{group_min}, {group_max}]")

        # for each variable of the given dataframe we plot a chart
        for var_name in cols:
            # we plot and save a chart for a single variable
            filename = var_name.replace(".", "").replace(
                ": ", "").replace("?", "").replace("/", "")
            folder_name = table_name.replace(
                ".", "").replace(": ", "").replace("?", "").replace("/", "")
            ChartServices.drawHistogram(
                df, var_name, f"Count_{var_name}", f"chart_{filename}", f"charts_{folder_name}")

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
        results, var_name, all_, group_values = DataManager.__initialize_results_dict(
            df, group)

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
                    group_totals[group_val] = df.loc[df[group]
                                                     == group_val, col].dropna().shape[0]

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
                        filter = (df[group] == group_val) & (
                            df[col] == category)

                        # We compute the statistics needed
                        sub_category_total = df[filter].shape[0]
                        sub_category_percent = round(
                            sub_category_total/(group_totals[group_val]) * 100, 2)
                        results[f"{group} {group_val}"].append(
                            f"{sub_category_total} ({sub_category_percent}%)")

                        # We save data for the charts
                        single_data_for_chart[group_val].append(
                            float(sub_category_total))

            data_for_chart.append(single_data_for_chart)

        # we make a chart from this analysis
        for item in data_for_chart:
            if group is not None:

                # plotting the chart
                filename = item["col_name"].replace(".", "").replace(
                    ": ", "").replace("?", "").replace("/", "")
                folder_name = table_name.replace(
                    ".", "").replace(": ", "").replace("?", "").replace("/", "")
                ChartServices.drawBinaryGroupedBarChart(
                    item["values"], {"label": "Male", "values": item["1.0"]},
                    {"label": "Female",
                        "values": item["0.0"]}, "Categories", "Count", item["col_name"],
                    f"chart_{filename}", f"charts_{folder_name}")

        # we return the data frame containing the informations
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
        table_df = table_df[table_df["Tag"] == "Phase 1"]
        table_df = table_df.drop(["Tag"], axis=1)

        # We get the dataframe from the table the table containing the sex information
        if "34500 Sex" in cols:
            sex_df = table_df[["Participant", "34500 Sex"]]
            table_df = table_df.drop(["34500 Sex"], axis=1)
        else:
            sex_df = self.get_table("General_1_Demographic Questionnaire", columns=["Participant", "Tag", "34500 Sex"])
            sex_df = sex_df[sex_df["Tag"] == "Phase 1"]
            sex_df = sex_df.drop(["Tag"], axis=1)

        # we retrieve categorical and numerical data
        categorical_df = Helpers.retrieve_categorical(table_df, ids=["Participant"])
        numerical_df = Helpers.retrieve_numerical(table_df, ids=["Participant"])

        # we merge the the categorical dataframe with the general dataframe by the column "Participant"
        categorical_df = pd.merge(sex_df, categorical_df, on="Participant", how="inner")
        categorical_df = categorical_df.drop(["Participant"], axis=1)

        # we merge the the numerical dataframe with the general dataframe by the column "Participant"
        numerical_df = pd.merge(sex_df, numerical_df, on="Participant", how="inner")
        numerical_df = numerical_df.drop(["Participant"], axis=1)

        # we make a categorical var analysis for this table
        categorical_stats = self.get_categorical_var_analysis(table_name, categorical_df, group="34500 Sex")

        # we make a numerical var analysis for this table
        numerical_stats = self.get_numerical_var_analysis(table_name, numerical_df, group="34500 Sex")

        # we concatenate all the results to get the final dataframe
        final_df = pd.concat(
            [categorical_stats, numerical_stats], ignore_index=True)
        filename = table_name.replace(".", "").replace(
            ": ", "").replace("?", "").replace("/", "")

        # if saveInFile True we save the dataframe in a csv file
        if save_in_file:
            Helpers.save_stats_file(filename, final_df)

        # we return the dataframe
        return final_df

    def get_variable_info(self, var_name):
        """
        Function that returns all the information about a specific variable

        :param var_name: The name of the variable
        :return: a python dictionary containing all the infos
        """
        # we extract the variable id from the variable name
        var_id = Helpers.extract_var_id(var_name)

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

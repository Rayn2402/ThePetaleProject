"""

Authors : Nicolas Raymond
          Mehdi Mitiche

This file contains all functions linked to SQL data management

"""
import SQLutils.ChartServices as ChartServices
import SQLutils.Helpers as Helpers
import psycopg2
import pandas as pd
import os
import csv
from pathlib import Path
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

        query = f"CREATE TABLE {self.schema}.\"{table_name}\" (" + Helpers.colsAndTypes(types)

        if primary_key is not None:

            # We define the primary key
            keys = Helpers.colsForSql(primary_key)
            query += f", PRIMARY KEY ({keys}) );"

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
            self.cur.copy_from(file, f"{self.schema}.\"{table_name}\"", sep="!", null=" ")
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

    def get_gender_stats(self):
        """
        Gets the count of all participant, the count of males, and the count of females from phase 01


        return: a list  of three numbers [#all, #female, #male]
        """

        # we create a data frame from the data in the table
        df = self.get_table("General_1_Demographic Questionnaire", [
                            "Participant", "Tag", "34500 Sex"])

        # we select only male participants from phase 01
        df_male = df[(df["Tag"] == "Phase 1") & (df["34500 Sex"] == "1.0")]

        # we select only female participants from phase 02
        df_female = df[(df["Tag"] == "Phase 1") & (df["34500 Sex"] == "0.0")]

        # we return the results
        return [df_male.shape[0] + df_female.shape[0], df_female.shape[0], df_male.shape[0]]

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
            all_mean = round(df[col].astype("float").mean(axis=0), 2)
            all_var = round(df[col].astype("float").var(axis=0), 2)
            results[all_].append(f"{all_mean} ({all_var})")

            # if the group is given, we calculate the stats for each possible value of that group
            if group is not None:
                for group_val in group_values:
                    # we append the mean and the var for sub group participants to the results dictionary
                    df_group = df[df[group] == group_val]
                    group_mean = round(
                        df_group[col].astype("float").mean(axis=0), 2)
                    group_var = round(
                        df_group[col].astype("float").var(axis=0), 2)
                    results[f"{group} {group_val}"].append(
                        f"{group_mean} ({group_var})")

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
        results, var_name, all_, group_values = DataManager.__initialize_results_dict(df, group)

        # we get the columns on which we will calculate the stats
        cols = [col for col in df.columns if col != group]

        # we initialize a python list where we will save data that will be useful when plotting the charts
        data_for_chart = []

        # for each column we calculate the count and the percentage
        for col in cols:
            # we initialize an object that will contain data that will be useful to plot this particular variable
            if group is None:
                single_data_for_chart = {
                    "col_name": col, "values": [], "all": []}
            else:
                single_data_for_chart = {
                    "col_name": col, "values": [], "all": []}
                for group_val in group_values:
                    single_data_for_chart[group_val] = []

            # we get all the categories of this variable
            categories = df[col].unique()

            # we get the total count
            total = df.shape[0]

            # for each category of this variable we get the counts and the percentage
            for category in categories:

                if category is not None:

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
                            df_group = df[df[group] == group_val]
                            sub_category_total = df_group[df_group[col]
                                                          == category].shape[0]
                            sub_category_percent = round(
                                sub_category_total/category_total * 100, 2)
                            results[f"{group} {group_val}"].append(
                                f"{category_total} ({sub_category_percent}%)")

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
                    {"label": "Female", "values": item["0.0"]}, "Categories", "Count", item["col_name"],
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

    def get_table_stats(self, table_name, conditions=[{"col": "Tag", "val": "Phase 1"}],
                        include="ALL", exclude=["Date", "Form", "Status", "Remarks"], save_in_file=True):
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
        for cond in conditions:
            table_df = table_df[table_df[cond["col"]] == cond["val"]]

        # we retrieve categorical data
        categorical_df = Helpers.retrieve_categorical(
            table_df, ids=["Participant"])

        # we retrieve numerical data
        numerical_df = Helpers.retrieve_numerical(
            table_df, ids=["Participant"])

        # we build a dataframe from the table the table containing the gender information
        df_general = self.get_table("General_1_Demographic Questionnaire", [
            "Participant", "Tag", "34500 Sex"])

        # we get only the rows that satisfy the given conditions
        for cond in conditions:
            df_general = df_general[df_general[cond["col"]] == cond["val"]]

        # we select only two columns Participant and 34500 Sex
        df_general = df_general[["Participant", "34500 Sex"]]

        # we merge the the categorical dataframe with the general dataframe by the column "Participant"
        categorical_df = pd.merge(
            categorical_df, df_general, on="Participant", how="inner")

        # we merge the the numerical dataframe with the general dataframe by the column "Participant"
        numerical_df = pd.merge(numerical_df, df_general,
                                on="Participant", how="inner")

        # we remove useless columns
        categorical_df = categorical_df[[
            col for col in categorical_df.columns if col not in ["Participant", "Tag"]]]
        numerical_df = numerical_df[[
            col for col in numerical_df.columns if col not in ["Participant", "Tag"]]]

        # we make a categorical var analysis for this table
        categorical_stats = self.get_categorical_var_analysis(table_name,
                                                              categorical_df, group="34500 Sex")
        # we make a numerical var analysis for this table
        numerical_stats = self.get_numerical_var_analysis(
            table_name, numerical_df, group="34500 Sex")

        # we concatenate all the results to get the final dataframe
        final_df = pd.concat(
            [categorical_stats, numerical_stats], ignore_index=True)

        filename = table_name.replace(
            ".", "").replace(": ", "").replace("?", "").replace("/", "")

        # if saveInFile True we save the dataframe in a csv file
        if save_in_file:
            if not os.path.exists(f"./stats/stats_{filename}"):
                Path(
                    f"./stats/stats_{filename}").mkdir(parents=True, exist_ok=True)
            if os.path.isfile(f"./stats/stats_{filename}/stats_{filename}.csv"):
                os.remove(f"./stats/stats_{filename}/stats_{filename}.csv")
            final_df.to_csv(f"./stats/stats_{filename}/stats_{filename}.csv")

        # we return the dataframe
        return final_df

    def get_generale_stats(self, save_in_file=True):
        """
        Function that returns a dataframe containing statistics from the generale Table

        :param save_in_file: Boolean, if true the dataframe will be saved in a csv file in the folder generale_stats
        :return: pandas DataFrame
        """

        # the variables for which we will calculate the statistics
        variables = [
            {"table_name": "General_2_CRF Hematology-Oncology",
             "var_name": "34472 Age at diagnosis", "type": 0},
            {"table_name": "General_2_CRF Hematology-Oncology",
             "var_name": "34471 Date of diagnosis", "type": 0},
            {"table_name": "General_2_CRF Hematology-Oncology",
             "var_name": "34474 Date of treatment end", "type": 0},
            {"table_name": "General_2_CRF Hematology-Oncology",
             "var_name": "34475 Risk group", "type": 1},
            {"table_name": "General_2_CRF Hematology-Oncology",
             "var_name": "34477 Boston protocol followed", "type": 1},
            {"table_name": "General_2_CRF Hematology-Oncology",
             "var_name": "34479 Radiotherapy?", "type": 1},
            {"table_name": "General_2_CRF Hematology-Oncology",
             "var_name": "34480 Radiotherapy dose", "type": 0},
            {"table_name": "General_1_Demographic Questionnaire",
             "var_name": "34502 Height", "type": 0},
            {"table_name": "General_1_Demographic Questionnaire",
             "var_name": "34503 Weight", "type": 0},
            {"table_name": "General_1_Demographic Questionnaire",
             "var_name": "34604 Is currently smoking?", "type": 1},
            {"table_name": "General_1_Demographic Questionnaire",
             "var_name": "34500 Sex", "type": 1}
        ]

        # we initialize the lists that will contain the name of the
        # categorical and numerical columns for both tables general_1 and general_2
        categorical_cols_1, categorical_cols_2 = [], []
        numerical_cols_1, numerical_cols_2 = [], []

        # we fill the lists with categorical and numerical columns for both tables general_1 and general_2
        for var in variables:
            if var["table_name"] == "General_1_Demographic Questionnaire":
                if var["type"] == 1:
                    categorical_cols_1.append(var["var_name"])
                else:
                    numerical_cols_1.append(var["var_name"])
            elif var["table_name"] == "General_2_CRF Hematology-Oncology":
                if var["type"] == 1:
                    categorical_cols_2.append(var["var_name"])
                else:
                    numerical_cols_2.append(var["var_name"])

        # we get a dataframe from the table general_1
        df_general_1 = self.get_table(
            "General_1_Demographic Questionnaire",  categorical_cols_1 + numerical_cols_1 + ["Participant", "Tag"])
        # we get a dataframe from the table general_2
        df_general_2 = self.get_table(
            "General_2_CRF Hematology-Oncology", categorical_cols_2 + numerical_cols_2 + ["Participant", "Tag"])

        # we add a new column to the table general_2 : Time of treatment
        df_general_2["Time of treatment"] = df_general_2["34474 Date of treatment end"] - \
                                            df_general_2["34471 Date of diagnosis"]
        # we transform the time of treatment to months
        df_general_2["Time of treatment"] = df_general_2["Time of treatment"].apply(
            Helpers.timeDeltaToMonths)

        # we get only survivors from Phase 1
        df_general_1 = df_general_1[df_general_1["Tag"] == "Phase 1"]
        df_general_2 = df_general_2[df_general_2["Tag"] == "Phase 1"]

        # we delete "34474 Date of treatment end", and "34471 Date of diagnosis" from the numerical columns of general_2 and we replace them by time of treatment
        numerical_cols_2.remove("34474 Date of treatment end")
        numerical_cols_2.remove("34471 Date of diagnosis")
        numerical_cols_2.append("Time of treatment")

        # we retrieve only the categorical columns for the table general_1
        categorical_data_1 = df_general_1[categorical_cols_1]
        # we retrieve only the numerical columns for the table general_1
        numerical_data_1 = df_general_1[numerical_cols_1 + ["34500 Sex"]]

        # we retrieve only the categorical columns for the table general_2
        categorical_data_2 = df_general_2[categorical_cols_2 + ["Participant"]]
        # we retrieve only the numerical columns for the table general_2
        numerical_data_2 = df_general_2[numerical_cols_2 + ["Participant"]]

        # we add the column "34500 Sex"to numerical_data_2 and categorical_data_2,  this is useful because we want to get the stats for each value  of this group
        categorical_data_2 = pd.merge(categorical_data_2, df_general_1,
                                      on="Participant", how="inner")
        categorical_data_2 = categorical_data_2[[
            col for col in categorical_data_2.columns if col in categorical_cols_2 + ["34500 Sex"]]]
        numerical_data_2 = pd.merge(numerical_data_2, df_general_1,
                                    on="Participant", how="inner")
        numerical_data_2 = numerical_data_2[[
            col for col in numerical_data_2.columns if col in numerical_cols_2 + ["34500 Sex"]]]

        # we make a categorical var analysis for general_1
        categorical_stats_1 = self.get_categorical_var_analysis("General_1_Demographic Questionnaire",
                                                                categorical_data_1, group="34500 Sex")
        # we make a numerical var analysis for general_1
        numerical_stats_1 = self.get_numerical_var_analysis(
            "General_1_Demographic Questionnaire", numerical_data_1, group="34500 Sex")

        # we make a categorical var analysis for general_2
        categorical_stats_2 = self.get_categorical_var_analysis("General_2_CRF Hematology-Oncology",
                                                                categorical_data_2, group="34500 Sex")
        # we make a numerical var analysis for general_2
        numerical_stats_2 = self.get_numerical_var_analysis(
            "General_2_CRF Hematology-Oncology", numerical_data_2, group="34500 Sex")

        # we concatenate all the results to get the final dataframe
        final_df = pd.concat(
            [categorical_stats_1, numerical_stats_1, categorical_stats_2, numerical_stats_2], ignore_index=True)

        filename = "General"
        # we save the dataframe in a csv file
        if save_in_file:
            if not os.path.exists(f"./stats/stats_{filename}"):
                Path(
                    f"./stats/stats_{filename}").mkdir(parents=True, exist_ok=True)
            if os.path.isfile(f"./stats/stats_{filename}/stats_{filename}.csv"):
                os.remove(f"./stats/stats_{filename}/stats_{filename}.csv")
            final_df.to_csv(f"./stats/stats_{filename}/stats_{filename}.csv")

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

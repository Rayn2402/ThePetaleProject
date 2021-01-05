"""

Authors : Nicolas Raymond

This file contains all functions linked to SQL data management

"""

import psycopg2
import pandas as pd
import os
import csv
import numbers
from pathlib import Path


import helpers
import chartServices


class DataManager:

    def __init__(self, user, password='petale101', database='petale', host='localhost', port='5437', schema='public'):
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

        return conn, cur

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
            # for col in columns:
            #    query = f"{query} \"{col}\""
            # Small Fix happened here
            columnsToFetch = helpers.colsForSql(columns)
            query = f"{query} {columnsToFetch}"

        # We add table name to the query
        query = f"{query} FROM {self.schema}.\"{table_name}\""

        # We execute the query
        try:
            self.cur.execute(query)

        except psycopg2.Error as e:
            print(e.pgerror)

        # We retrive column names and data
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
        # we execute the query
        try:
            self.cur.execute(
                f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG='petale' AND table_schema = '{self.schema}' ")
        except psycopg2.Error as e:
            print(e.pgerror)
        tables = self.cur.fetchall()

        # We reset the cursor
        self.reset_cursor()

        # we return the result
        return list(map(lambda t: t[0], tables))

    def get_missing_data_count(self, tableName, drawChart=False, excludedCols=["Remarks"]):
        """
        get the count of all the missing data of one given table

        :param tableName: name of the table
        :param drawChart: boolean to indicate if a chart should be created to visulize the missing data
        :param: excludeCols: list of strings containing the list of columns to exclude
        :return: a python dictionary containing the missing data count and the number of complete rows

        """

        # Extracting the name of the columns of the given  table
        escapedTableName = tableName.replace("'", "''")
        try:
            self.cur.execute(
                f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME= \'{escapedTableName}\'")
        except psycopg2.Error as e:
            print(e.pgerror)
        columns_y = []
        cols = self.cur.fetchall()
        columns_y = list(map(lambda c: c[0], cols))

        # Excluding the none needed columns
        columns_y = [col for col in columns_y if col not in excludedCols]

        missing_x = [0] * len(columns_y)

        # Get the cols ready to be in the SQL query
        columnsToFetch = helpers.colsForSql(columns_y)

        # we execute the query
        try:
            self.cur.execute(
                f'SELECT {columnsToFetch} FROM \"{tableName}\" ')
        except psycopg2.Error as e:
            print(e.pgerror)
        rows = self.cur.fetchall()
        # we intialize the counters
        missingCount = 0
        completedRowCount = 0

        for r in rows:
            completedRow = True
            for index, item in enumerate(r):
                if(item == None):
                    completedRow = False
                    missingCount += 1
                    missing_x[index] += 1
            if(completedRow == True):
                completedRowCount += 1

        # Plotting the bar chart
        if(drawChart):
            fileName = "missing_data_" + \
                tableName.replace(".", "").replace(":", "")
            folderName = "missing_data_charts"
            figureTitle = f'Count of missing data by columns names for the table {tableName}'
            chartServices.drawBarhChart(
                columns_y, missing_x, "Columns", "Data missing", figureTitle, fileName, folderName)

        # We reset the cursor
        self.reset_cursor()

        # returning a dictionary containing the data needed
        return {"tableName": tableName, "missingCount": missingCount, "completedRowCount": completedRowCount, "totalRows": len(rows)}

    def get_all_missing_data_count(self, filename="petale_missing_data.csv", drawCharts=True):
        """
        Function that generate a csv file containing the count of the missing data of all the tables of the database

        :param filename: the name of file to be genrated
        :param foldername: the name of the folder where the file will be created
        :param drawChart:  boolean to indicate if a chart should be created to visulize the missing data of each table

        :generate a csv file


        """
        # we initalize the results
        results = []

        # we get all the table names
        tables = self.getAllTables()
        length = len(tables)

        # For each table we get the missing data count
        for index, table in enumerate(tables):
            print(f"Processing data... {index}/{length}")
            missingDataCount = self.getMissingDataCount(table, drawCharts)

            # we save the missing data count in results
            results.append(missingDataCount)

        # we generate a csv file from the data in results
        helpers.writeCsvFile(results, filename)
        print("File is ready in the folder missing_data! ")

    def get_common_count(self, tables, columns=["Participant", "Tag"], saveInFile=False):
        """
        get the number of common survivors from a list of tables

        :param tebles: the list of tables
        :param columns: list of the columns according to we want to get the the common survivors
        :return: number of common survivors

        """
        # we prepare the columns to be in the SQL query
        colsInQuery = helpers.colsForSql(columns)

        # we build the request
        query = 'SELECT COUNT(*) FROM ('
        for index, table in enumerate(tables):
            if(index == 0):
                query += f'(SELECT {colsInQuery} FROM \"{table}\")'
            else:
                query += f' INTERSECT (SELECT "Participant","Tag" FROM \"{table}\")'
        query += ') l'

        # We execute the query
        try:
            self.cur.execute(query)
        except psycopg2.Error as e:
            print(e.pgerror)

        row = self.cur.fetchall()[0]

        if(saveInFile == True):
            # Saving in file
            if(os.path.isfile("cummonPatients.csv") == False):
                try:
                    with open("cummonPatients.csv", 'w', newline='') as csvfile:
                        writer = csv.DictWriter(
                            csvfile, ["tables", "cummonSurvivors"])
                        writer.writeheader()
                        separator = " & "
                        writer.writerow({"tables": separator.join(
                            tables), "cummonSurvivors": row[0]})
                except IOError:
                    print("I/O error")
            else:
                try:
                    # Open file in append mode
                    with open("cummonPatients.csv", 'a+', newline='') as csvfile:
                        # Create a writer object from csv module
                        writer = csv.DictWriter(
                            csvfile, ["tables", "cummonSurvivors"])
                        # Add a new row to the csv file
                        separator = " & "
                        writer.writerow({"tables": separator.join(
                            tables), "cummonSurvivors": row[0]})
                except IOError:
                    print("I/O error")
        return row[0]

    def get_gender_stats(self):
        """
        get the count of all participant, the count of males, and the count of females from phase 01


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

    def get_numirical_var_analysis(self, table_name, var_name):
        """
        Calculate the mean and variance for All, Male, and Female survivors  of a given numirical variable in a given table

        :param table_name: name of the table
        :param cvar_name: name of the variable
        return: a list  of three dictionary, each dictionary contains two keys : mean and var [dict_all, dict_female, dict_male]
        """
        # if the given table is diffrent from the table general_1 we merge the two tables
        if(table_name != "General_1_Demographic Questionnaire"):
            # we build a dataframe from the table the table containing the gender information
            df_1 = self.get_table("General_1_Demographic Questionnaire", [
                "Participant", "Tag", "34500 Sex"])

            # we extract only survivors from Phase 1
            df_1 = df_1[df_1["Tag"] == "Phase 1"]

            # we select only two columns Participant and 34500 Sex
            df_1 = df_1[["Participant", "34500 Sex"]]

            # We get a dataframe of the given table with tha variables that wee need
            if(var_name != "Time of treatment"):
                cols = ["Participant", "Tag", var_name]
            # in the case of the variable time of treatment we have to slect two more cols : Date at diagnosis and date of treatment end
            else:
                cols = ["Participant", "Tag", "34471 Date of diagnosis",
                        "34474 Date of treatment end"]
            try:
                df_2 = self.get_table(table_name, cols)
            except:
                print(
                    "Error occured, please check if the tableName and the varName are correct")
                return [None, None, None]

            # we extract only survivors from Phase 1
            df_2 = df_2[df_2["Tag"] == "Phase 1"]

            # we select only two columns Participant and the Variable we want
            if(var_name != "Time of treatment"):
                df_2 = df_2[["Participant", var_name]]
            # in the case of the variable Time of treatment we create a new column : Time of treatment = date of the end of treatment - date of diagnosis
            else:
                # we calculate the time of treatment
                df_2["Time of treatment"] = df_2["34474 Date of treatment end"] - \
                    df_2["34471 Date of diagnosis"]
                # we transform the time of treatment to months
                df_2["Time of treatment"] = df_2["Time of treatment"].apply(
                    helpers.timeDeltaToMonths)
                # we select only two columns Participant and the time of treatment
                df_2 = df_2[["Participant", var_name]]
            # we merge the two dataframes by the column "Participant"
            df = pd.merge(df_2, df_1, on="Participant", how="inner")
        # we work directly in the table general_1
        else:
            # We get a dataframe of the given table with tha variables that wee need
            try:
                df = self.get_table("General_1_Demographic Questionnaire", [
                    "Participant", "Tag", "34500 Sex", var_name])
            except:
                print(
                    "Error occured, please check if the tableName and the varName are correct")
                return [None, None, None]

            # we extract only survivors from Phase 1
            df = df[df["Tag"] == "Phase 1"]

            # we select only the columns Participant and 34500 Sex and given variable
            df = df[["Participant", "34500 Sex", var_name]]

        # we extract the only the male survivors
        df_male = df[df["34500 Sex"] == "1.0"]

        # we extract the only the female survivors
        df_female = df[df["34500 Sex"] == "0.0"]

        # we get the stats(mean, variance) of all survivors
        all_stats = {"mean": round(df[var_name].astype("float").mean(
            axis=0), 2), "var": round(df[var_name].astype("float").var(axis=0), 2)}

        # we get the stats(mean, variance) of Male survivors
        male_stats = {"mean": round(df_male[var_name].astype("float").mean(
            axis=0), 2), "var": round(df_male[var_name].astype("float").var(axis=0), 2)}

        # we get the stats(mean, variance) of Female survivors
        female_stats = {"mean": round(df_female[var_name].astype("float").mean(
            axis=0), 2), "var": round(df_female[var_name].astype("float").var(axis=0), 2)}

        # Plot Chart Comming soon
        filename = var_name.replace(".", "").replace(": ", "").replace("?", "")
        folder_name = table_name.replace(
            ".", "").replace(": ", "").replace("?", "")
        chartServices.drawHistogram(
            df, "Values", "Count", var_name, f"chart_{filename}", f"charts_{folder_name}")

        # we return the results
        return [all_stats, female_stats, male_stats]

    def get_categorical_var_analysis(self, table_name, var_name):
        """Calculate the counts of all the category of this variable for All, Female, and male survivors

        :param table_name: name of the table
        :param cvar_name: name of the variable
        return: a pandas dataframe
        """

        if(table_name != "General_1_Demographic Questionnaire"):
            # we build a dataframe from the table the table containing the gender information
            df_1 = self.get_table("General_1_Demographic Questionnaire", [
                "Participant", "Tag", "34500 Sex"])

            # we extract only survivors from Phase 1
            df_1 = df_1[df_1["Tag"] == "Phase 1"]

            # we select only two columns Participant and 34500 Sex
            df_1 = df_1[["Participant", "34500 Sex"]]

            # We get a dataframe of the given table with tha variables that wee need
            cols = ["Participant", "Tag", var_name]
            try:
                df_2 = self.get_table(table_name, cols)
            except:
                print(
                    "Error occured, please check if the tableName and the varName are correct")
                return [None, None, None]

            # we extract only survivors from Phase 1
            df_2 = df_2[df_2["Tag"] == "Phase 1"]

            # we select only two columns Participant and the Variable we want
            df_2 = df_2[["Participant", var_name]]

            # we merge the two dataframes by the column "Participant"
            df = pd.merge(df_2, df_1, on="Participant", how="inner")
        # we work directly in the table general_1
        else:
            # We get a dataframe of the given table with tha variables that wee need
            try:
                df = self.get_table("General_1_Demographic Questionnaire", [
                    "Participant", "Tag", "34500 Sex", var_name])
            except:
                print(
                    "Error occured, please check if the tableName and the varName are correct")
                return [None, None, None]

            # we extract only survivors from Phase 1
            df = df[df["Tag"] == "Phase 1"]

            # we select only the columns Participant and 34500 Sex and given variable
            df = df[["Participant", "34500 Sex", var_name]]

        # we extract the only the male survivors
        df_male = df[df["34500 Sex"] == "1.0"]

        # we extract the only the female survivors
        df_female = df[df["34500 Sex"] == "0.0"]

        # we get all the possible values of this variable
        values = df[var_name].unique()

        dict = {}
        # For each value we save the count of this value in all, female, and male survivors
        for val in values:
            if(val != None):
                dict[val] = [df[df[var_name] == val].shape[0], df_female[df_female[var_name]
                                                                         == val].shape[0], df_male[df_male[var_name] == val].shape[0]]
        # if(df[df[var_name].isnull()].shape[0] != 0):
        #    dict["null"] = [df[df[var_name].isnull()].shape[0], df_female[df_female[var_name].isnull(
        #    )].shape[0], df_male[df_male[var_name].isnull()].shape[0]]

        # we make a chart from this analysis

        # Preparing the data to plot chart
        data_male = {"label": "Male", "values": []}
        data_female = {"label": "Female", "values": []}

        for key in dict.keys():
            data_male["values"].append(float(dict[key][2]))
            data_female["values"].append(float(dict[key][1]))

        # ploting the chart
        filename = var_name.replace(".", "").replace(": ", "").replace("?", "")
        folder_name = table_name.replace(
            ".", "").replace(": ", "").replace("?", "")
        chartServices.drawBinaryGroupedBarChart(
            dict.keys(), data_male, data_female, "Categories", "Count", var_name, f"chart_{filename}", f"charts_{folder_name}")

        # we return the data frame containing the informations
        return pd.DataFrame(dict)

    def get_generale_stats(self, save_in_file=True):
        """
        Function that return a dataframe containing statistics from the generale Table

        :param saveInFile: Boolean, if true the dataframe will be saved in a csv file in the folder generale_stats
        :return: pandas DataFrame
        """

        # the variables for which we will calculate the statistics
        sources = [
            {"table_name": "General_2_CRF Hematology-Oncology",
                "var_name": "34472 Age at diagnosis", "type": 0},
            {"table_name": "General_2_CRF Hematology-Oncology",
                "var_name": "Time of treatment", "type": 0},
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
        ]

        # we set up a python dictionary that will contain the results
        result = {"variable": [], "All Survivors": [],
                  "Male": [], "Female": []}

        # we get the gender stats
        nb, nb_female, nb_male = self.get_gender_stats()

        # we save the gender stats in the results dictionary
        result["variable"].append("Number of survivors")
        result["All Survivors"].append(nb)
        result["Female"].append(nb_female)
        result["Male"].append(nb_male)

        # for each variable we calculate the stats and we save it in results
        for source in sources:
            # if type 0, its a numirical variable, so we use getNumiricalVarAnalysis
            if(source["type"] == 0):
                all_survivors, female_survivors, male_survivors = self.get_numirical_var_analysis(
                    source["table_name"], source["var_name"])
                result["variable"].append(source["var_name"])
                result["All Survivors"].append(
                    f"{all_survivors['mean']} ({all_survivors['var']})")
                result["Female"].append(
                    f"{female_survivors['mean']} ({female_survivors['var']})")
                result["Male"].append(
                    f"{male_survivors['mean']} ({male_survivors['var']})")
            # if type 1, its a categorical variable, so we use getCategoricalVarAnalysis
            else:
                df = self.get_categorical_var_analysis(
                    source["table_name"], source["var_name"])
                for col in df.columns:
                    result["variable"].append(f"{source['var_name']} : {col}")
                    result["All Survivors"].append(
                        f"{df[col][0]} (100%)")
                    percent_female = round(df[col][1]/df[col][0] * 100, 2)
                    result["Female"].append(
                        f"{df[col][1]} ({percent_female}%)")
                    percent_male = round(df[col][2]/df[col][0] * 100, 2)
                    result["Male"].append(
                        f"{df[col][2]} ({percent_male}%)")
        # we create the dataframe
        df = pd.DataFrame(result)

        # if saveInFile True we save the dataframe in a csv file
        if(save_in_file == True):
            if not os.path.exists("./stats/stats_general"):
                Path(f"./stats/stats_general").mkdir(parents=True, exist_ok=True)
            if(os.path.isfile("./stats/stats_general/stats_general.csv") == True):
                os.remove("./stats/stats_general/stats_general.csv")
            df.to_csv("./stats/stats_general/stats_general.csv")

        return df

    def get_table_stats(self, table_name, include="ALL", exclude=[], save_in_file=True):
        # we get a dataframe of the given tablename
        if(include == "ALL"):
            table_df = self.get_table(table_name)
        else:
            table_df = self.get_table(table_name, include)

        # we retreive the columns of the table
        cols = table_df.columns

        # we select only the columns with the name in this format "nnnnn VariableName" ex : 35428 niveau d'activité physique(temps)
        cols = [col for col in cols if col.split()[0].isnumeric()]

        # we exclude the parameters specified with the exclude parameter
        cols = [col for col in cols if col not in exclude]

        # we set up a python dictionary that will contain the results
        result = {"variable": [], "All Survivors": [],
                  "Male": [], "Female": []}

        # for each columns we analyse the variable
        for col in cols:
            # we check if the the variable is categorical or numirical
            if(helpers.check_categorical_var(table_df[col])):
                # Categorical variable analysis
                df_categories = self.get_categorical_var_analysis(
                    table_name, col)
                for catg in df_categories.columns:
                    # we save the results
                    result["variable"].append(f"{col} : {catg}")
                    result["All Survivors"].append(
                        f"{df_categories[catg][0]} (100%)")
                    percent_female = round(
                        df_categories[catg][1]/df_categories[catg][0] * 100, 2)
                    result["Female"].append(
                        f"{df_categories[catg][1]} ({percent_female}%)")
                    percent_male = round(
                        df_categories[catg][2]/df_categories[catg][0] * 100, 2)
                    result["Male"].append(
                        f"{df_categories[catg][2]} ({percent_male}%)")
            else:
                # Numirical variable analysis
                all_survivors, female_survivors, male_survivors = self.get_numirical_var_analysis(
                    table_name, col)
                # we save the results
                result["variable"].append(col)
                result["All Survivors"].append(
                    f"{all_survivors['mean']} ({all_survivors['var']})")
                result["Female"].append(
                    f"{female_survivors['mean']} ({female_survivors['var']})")
                result["Male"].append(
                    f"{male_survivors['mean']} ({male_survivors['var']})")

        # we create the dataframe from results
        result_df = pd.DataFrame(result)
        # if saveInFile True we save the dataframe in a csv file

        filename = table_name.replace(
            ".", "").replace(": ", "").replace("?", "")

        if(save_in_file == True):
            if not os.path.exists(f"./stats/stats_{filename}"):
                Path(
                    f"./stats/stats_{filename}").mkdir(parents=True, exist_ok=True)
            if(os.path.isfile(f"./stats/stats_{filename}/stats_{filename}.csv") == True):
                os.remove(f"./stats/stats_{filename}/stats_{filename}.csv")
            result_df.to_csv(f"./stats/stats_{filename}/stats_{filename}.csv")

        # we return the dataframe
        return result_df

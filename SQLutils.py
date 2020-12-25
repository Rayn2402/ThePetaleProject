"""

Authors : Nicolas Raymond

This file contains all functions linked to SQL data management

"""

import psycopg2
import pandas as pd
import os
import csv


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
            for col in columns:
                query = f"{query} \"{col}\""

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

    def getAllTables(self):
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

    def getMissingDataCount(self, tableName, drawChart=False, excludedCols=["Remarks"]):
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
            fileName = "missing_data_" + tableName.replace(".", "").replace(":","")
            folderName="missing_data_charts"
            figureTitle=f'Count of missing data by columns names for the table {tableName}'
            chartServices.drawBarhChart(columns_y,missing_x,"Columns","Data missing",figureTitle,fileName,folderName)
            
        # We reset the cursor
        self.reset_cursor()

        # returning a dictionary containing the data needed
        return {"tableName": tableName, "missingCount": missingCount, "completedRowCount": completedRowCount, "totalRows": len(rows)}

    def getAllMissingDataCount(self, filename, drawCharts=True):
        """
        Function that generate a csv file containing the count of the missing data of all the tables of the database

        :param filename: the name of file to be genrated 
        :param foldername: the name of the folder where the file will be created
        :param drawChart:  boolean to indicate if a chart should be created to visulize the missing data of each table

        :generate a csv file


        """
        #we initalize the results
        results = []

        #we get all the table names
        tables = self.getAllTables()
        length = len(tables)

        #For each table we get the missing data count
        for index, table in enumerate(tables):
            print(f"Processing data... {index}/{length}")
            missingDataCount = self.getMissingDataCount(table, drawCharts)

            #we save the missing data count in results
            results.append(missingDataCount)
            
        #we generate a csv file from the data in results
        helpers.writeCsvFile(results, filename)
    
    def getCommonCount(self, tables, columns = ["Participant","Tag"],saveInFile=False):
        """
        get the number of common survivors from a list of tables

        :param tebles: the list of tables
        :param columns: list of the columns according to we want to get the the common survivors
        :return: number of common survivors

        """
        #we prepare the columns to be in the SQL query
        colsInQuery = helpers.colsForSql(columns)

        #we build the request
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

        if(saveInFile==True):
            #Saving in file
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
                        writer = csv.DictWriter(csvfile,["tables", "cummonSurvivors"])
                        # Add a new row to the csv file
                        separator = " & "
                        writer.writerow({"tables": separator.join(
                            tables), "cummonSurvivors": row[0]})
                except IOError:
                    print("I/O error")
        return row[0]
    



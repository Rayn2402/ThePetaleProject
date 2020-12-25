"""

Authors : Nicolas Raymond

This file contains all functions linked to SQL data management

"""

import psycopg2
import pandas as pd
from matplotlib import pyplot as plt
import os


import helpers


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
        self.cur.execute(
            f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG='petale' AND table_schema = '{self.schema}' ")
        tables = self.cur.fetchall()

        # We reset the cursor
        self.reset_cursor()

        # we return the result
        return list(map(lambda t: t[0], tables))

    def getMissingDataCount(self, tableName, drawChart=False, excludedCols=["Remarks"]):
        """
        get the count of all the missing data of one given table

        :param tableName: name of the table
        :param drawChart: boolean to indicated if chart should be created from the data
        :param: excludeCols: list of strings containing the list of columns to exclude 
        :return: a python dictionary containing the missing data count and the number of complete rows

        """

        # Extracting the name of the columns of the given  table
        escapedTableName = tableName.replace("'", "''")
        self.cur.execute(
            f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME= \'{escapedTableName}\'")
        columns_x = []
        cols = self.cur.fetchall()
        columns_x = list(map(lambda c: c[0], cols))

        # Excluding the none needed columns
        columns_x = [col for col in columns_x if col not in excludedCols]

        missing_y = [0] * len(columns_x)

        # Get the cols ready to be in the SQL query
        columnsToFetch = helpers.colsForSql(columns_x)

        # we execute the query
        self.cur.execute(
            f'SELECT {columnsToFetch} FROM \"{tableName}\" ')
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
                    missing_y[index] += 1
            if(completedRow == True):
                completedRowCount += 1

        # Plotting the bar chart
        if(drawChart):
            # specifying the figure size
            fig = plt.figure(figsize=(40, .7*len(columns_x)))

            # specifying the type of the chart and the data in it
            plt.barh(columns_x, missing_y, color="#874bf2")

            # specifying the labels of the chart
            plt.ylabel('Columns')
            plt.xlabel('Data missing')

            # specifying the title of the chart
            plt.title(
                f'Count of missing data by columns names for the table {tableName}', fontsize=15)

            # saving the chart in a file in teh folder missing_data_charts
            if not os.path.exists('missing_data_charts'):
                os.makedirs('missing_data_charts')
            escapedName = tableName.replace(".", "")
            plt.savefig(
                f'./missing_data_charts/missing_data_{escapedName}.png')

            plt.close(fig)
        # We reset the cursor
        self.reset_cursor()

        # returning a dictionary containing the data needed
        return {"tableName": tableName, "missingCount": missingCount, "completedRowCount": completedRowCount, "totalRows": len(rows)}

    def getAllMissingDataCount(self, filename, drawCharts=True):
        results = []
        tables = self.getAllTables()
        length = len(tables)
        for index, table in enumerate(tables):
            print(f"Processing data... {index}/{length}")
            missingDataCount = self.getMissingDataCount(table, drawCharts)
            results.append(missingDataCount)
        helpers.writeCsvFile(results, filename)


manager = DataManager("mitm2902")

print(manager.getAllMissingDataCount("petale.png"))

"""

Authors : Nicolas Raymond

This file contains all functions linked to SQL data management

"""

import psycopg2
import pandas as pd


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
        self.conn, self.cur = DataManager.connect(user, password, database, host, port)
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
                query = f"{query} {col}"

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




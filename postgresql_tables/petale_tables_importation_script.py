"""

Author : Nicolas Raymond

This script creates a postgresql table for xlsx file containing data

"""

import time
import os
import csv
import argparse
import psycopg2
import pandas as pd
import shutil


# Initialization of Argument Parser
parser = argparse.ArgumentParser(description='Create table for each .xlsx file containing data')

# Required arguments
required = parser.add_argument_group('required arguments')
required.add_argument('--password', type=str, help='Postgresql username password', required=True)

# Optional arguments
parser.add_argument('--directory', default='./csv', type=str,
                    help='Directory where csv are temporary stored (default: ./csv)')
parser.add_argument('--user', default='postgres', type=str, help='Postgresql username (default: postgres)')
parser.add_argument('--dbname', default='postgres', type=str, help='Postgresql database (default: postgres)')
parser.add_argument('--host', default='localhost', type=str, help='Postgresql host (default: localhost)')
parser.add_argument('--schema', default='public', type=str, help='Postgresql schema (default: public)')
parser.add_argument('--port', default='5432', type=str, help="Port on which the database is connected (default: 5432)")

# Initialization of a conversion map
conversion_map = {"float": "numeric", "list (text)": "text", "text": "text", "integer": "numeric",
                  "text (long)": "text", "date": "date"}

delimiter = "*"
max_title_length = 60
meta_data_columns = {'Test ID': 'integer', 'Editorial board': 'text', 'Form': 'text', 'Number': 'text',
                     'Section': 'text', 'Test': 'text', 'Description': 'text', 'Type': 'text', 'Option': 'text',
                     'Unit': 'text', 'Remarks': 'text'}


# ############################################# TABLE CREATION FUNCTIONS ##############################

def create_table(conn, cur, schema, table_name, types):

    """
    Creates a table named "table_name" that as columns "columns" of type text.

    :param conn: table connection with psycopg2
    :param cur: cursor associated to the table
    :param schema: name of the schema
    :param table_name: name of the table
    :param types: names of the columns (key) and their respective types (value) in a dict
    """

    query = f"CREATE TABLE {schema}.\"{table_name}\" (\"Date\" date, \"Participant\" text, \"Form\" text, " \
        f"\"Day of Study\" numeric, \"Status\" text, \"Tag\" text, \"Remarks\" text ,"

    for col in types:
        query += f"\"{col}\" {types[col]}, "

    # We define the composite key
    query += f"PRIMARY KEY (\"Participant\", \"Tag\") );"

    cur.execute(query)
    conn.commit()


def fill_table(conn, cur, file_path, schema, table_name, delimiter, skip_header=True):

    """
    Fill a postgresql table with data from a csv file.

    :param conn: table connection with psycopg2
    :param cur: cursor associated to the table
    :param file_path: path of the csv file
    :param schema: database schema
    :param table_name: name of the table
    :param delimiter: character separating values in the csv file
    :param skip_header: bool indicating if we skip the first line of csv

    """

    # We open the file containing the data
    file = open(file_path, mode="r", newline="\n")

    # We skip the first line if necessary
    if skip_header:
        file.readline()

    # We copy the data to the table
    cur.copy_from(file, f"{schema}.\"{table_name}\"", sep=delimiter, null=" ")
    conn.commit()


def connect(user, password, dbname, host, port):

    """
    Creates a connection to the database

    :param user: username
    :param password: password
    :param dbname: database
    :param host: host
    :param port: port
    :return: connection and cursor
    """

    conn = psycopg2.connect(database=dbname,
                            user=user,
                            host=host,
                            password=password,
                            port=port)
    cur = conn.cursor()

    return conn, cur


def get_column_names(path):

    """
    Gets column names from first row of csv file.
    Commas are replaced by nothing

    :param path: csv file path
    :return: list with column names
    """

    with open(path, mode="r", newline="\n", ) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        columns = next(reader)

    return columns[7:]


def get_meta_data(conn, cur, schema):

    """
    Retrieves meta data file in a panda table
    Create a table for the meta data in the postgreSQL database

    :return: pandas dataframe
    """

    # We read the excel file
    df = pd.read_excel('191125_PETALE_meta_data.xlsx', engine='openpyxl')

    # We delete the aditionnal "Unnamed" column
    columns = [col for col in df.columns.values if col in meta_data_columns.keys()]
    df = df[columns]

    # We remove newlines in the "Option" column
    temp = df.Option[df.Option.notnull()]
    df.Option[df.Option.notnull()] = temp.apply(lambda x: x.replace("\n", "-"))

    # We save the file as csv in the same directory as the other file
    csv_path = os.path.join(args.directory, "PETALE_meta_data.csv")
    table_name = 'PETALE_meta_data'
    df.to_csv(csv_path, index=False, na_rep=" ", sep="!")

    # We create a table for the meta data
    query = f"CREATE TABLE {schema}.\"{table_name}\" ("

    for col in df.columns.values:
        query += f"\"{col}\" {meta_data_columns[col]}, "

    query = query[0:-2] + ");"

    cur.execute(query)
    conn.commit()

    # We fill the table
    fill_table(conn, cur, csv_path, schema, table_name, delimiter="!")

    return df


def split_column_name(column, units_list, sections, separator=" : "):
    """
    Split the column names in two part. Section and Test

    :param column: original column name
    :param units_list: list of measure unit contained in meta data file
    :param sections: list of sections contained in the meta data file
    :param separator: string separation pattern
    :return: section, part and unit (strings)
    """

    section, test = separate_section(column, sections, separator)
    test, unit = separate_unit(test, units_list)

    return section, test, unit


def separate_unit(test, units_list):
    """
    Separates the measure unit from the test name.
    Looks first if there is words in parenthesis at the end of the column name and
    then look if it's a measure unit or not.

    :param test: test name (strings)
    :param units_list: list of measure unit contained in meta data file
    :return: test, unit (string)
    """
    parts_list = test.split(" ")

    # We strip the string if necessary
    if parts_list[-1] == "":
        parts_list = parts_list[:-1]

    last_part = parts_list[-1]
    unit = ""

    # Their was a space in the potential unit
    if last_part[0] != "(" and last_part[-1] == ")":
        last_part = f"{parts_list[-2]} {parts_list[-1]}"

    if last_part[0] == "(" and last_part[-1] == ")":
        potential_unit = last_part[1:-1]
        if potential_unit in units_list:
            unit = potential_unit
            test = test[0:-(len(unit)+3)]

    return test, unit


def find_match(form, section, test, unit, meta_data):

    """
    Find the matching test in the meta data table

    :param form: value associated to column form
    :param section: value associated to column section
    :param test: value associated to column test
    :param unit: measure unit associated to the column (variable)
    :param meta_data: pandas dataframe containing meta data
    :return: new column name, type (strings)

    """

    row = meta_data[(meta_data['Form'] == form) & (meta_data['Section'] == section) & ((meta_data['Test'] == test) | (meta_data['Test'] == f"{test} ({unit})"))]
    if len(row) > 1:
        row = row[(row['Unit'] == unit)]

    type = conversion_map[row['Type'].values[0]]
    short_name = f"{row['Test ID'].values[0]} {row['Test'].values[0]}"

    if len(short_name) > max_title_length:
        short_name = f"{short_name[:max_title_length-3]}..."

    return short_name, type


def get_form(file_name):

    """
    Retrieves form from csv file name.

    :param file_name: csv file name
    :return: form (string)
    """
    # We want to remove the first part of csv file name delimited by a "_"
    parts_list = os.path.splitext(file_name)[0].split("_")

    # The remaining elements of the csv file name could have been separated due to "_" inserted to replace "/"
    # in measure unit when creating csv from excel
    form = concatenate(parts_list[2:], separator="/")

    return form


def concatenate(strings, separator=" "):
    """
    Concatenates all strings in a list while inserting "separator" betweem them.

    :param strings: list of strings
    :param separator: separator to insert between each string
    :return: string
    """
    result_string = strings[0]

    for s in strings[1:]:
        result_string += f"{separator}{s}"

    return result_string


def separate_section(column, sections, separator):

    """
    Retrieves the section name and the test name from the column title.

    :param column: column name (string)
    :param sections : list of sections contained in the meta data file
    :param separator: string separator
    :return: section, test (str, str)
    """
    # Splits the column name using separator
    parts_list = column.split(separator)

    # We validate the section name
    i = 3
    section = concatenate(parts_list[:min(i+1, len(parts_list))], separator)

    while section not in sections:
        section = concatenate(parts_list[:i], separator)
        i -= 1

    test = concatenate(parts_list[i+1:], separator)

    return section, test


# ################################ TEMPORARY CSV CREATION FUNCTIONS ########################################

def save_as_csv(folder, file_path, destination, index):

    """
    Saves the xlsx file "file path" as a csv and store it in a folder created for the file in
    the destination directory.

    :param folder: name of the folder containing the original file
    :param file_path: path of file to save as csv
    :param destination: destination directory
    :param index: rank of file in the subfolder
    """

    # We read the excel file
    df = pd.read_excel(file_path, engine='openpyxl')

    # We remove rows where "Participant" value is null
    df = df[~df.Participant.isnull()]

    # We fill that NaN value in "Tag" column
    df.Tag = df.Tag.fillna("Phase 1")

    file_name = f"{folder}_{index}_{df['Form'].values[0]}.csv".replace("/", "_")
    df.to_csv(os.path.join(destination, file_name), index=False, na_rep=" ", sep=delimiter)


def get_all_csv(args):

    """
    Saves all xlsx files contained in the subfolders of the directory

    :param args: arguments from arg parser
    """

    print("\nCreation of temporary CSVs..")
    start = time.time()

    # We retrieve destination directory to store csv files
    destination_dir = args.directory

    if not os.path.isdir(destination_dir):
        os.mkdir(destination_dir)

    # We create a list of all subfolders
    subfolders = [f.path for f in os.scandir() if f.is_dir()]
    subfolders = [sub for sub in subfolders if sub not in ['./.idea', './Raw files', './__pycache__', destination_dir]]

    # We create a csv for each xlsx files stored in the subfolders
    for sub in subfolders:
        folder = sub.split(os.path.sep)[1]
        for index, file in enumerate(os.listdir(sub)):
            file_path = os.path.join(sub, file)
            save_as_csv(folder, file_path, destination_dir, index)

    print('\n', 'DONE!', '\n')
    print('Execution Time: ', time.time() - start, "\n")


# ####################################### SCRIPT ############################################

if __name__ == '__main__':

    args = parser.parse_args()

    # We connect to the database
    conn, cur = connect(args.user, args.password, args.dbname, args.host, args.port)

    # We create temporary csv file from which the tables are built
    get_all_csv(args)

    # We connect to the database
    start = time.time()
    print("\nCreation of Tables..")

    # We list absolute paths of csv files in the directory
    files_path = [os.path.abspath(x) for x in os.listdir(args.directory)]
    files_path = [path for path in files_path if os.path.splitext(path)[1] == '.csv']

    # We retrieve meta data
    meta_data = get_meta_data(conn, cur, args.schema)
    meta_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # We save possible units
    units = meta_data.Unit.unique()
    sections = meta_data.Section.unique()

    # We change current directory
    os.chdir(args.directory)

    for path in files_path:

        # We get the columns name
        file_name = os.path.split(path)[1]
        form = get_form(file_name)
        columns = get_column_names(file_name)

        # We figure out the new column names and their types
        types = {}
        for i, c in enumerate(columns):
            section, test, unit = split_column_name(c, units, sections)
            name, type = find_match(form, section, test, unit, meta_data)
            types[name] = type

        # We create the table
        table_name = os.path.splitext(file_name)[0]
        if len(table_name) > max_title_length:
            table_name = table_name[0:max_title_length-3]+"..."

        create_table(conn, cur, args.schema, table_name, types)

        # We fill it
        fill_table(conn, cur, file_name, args.schema, table_name, delimiter)

    # We delete csv files created during the process
    os.chdir(os.path.dirname(os.getcwd()))
    shutil.rmtree(args.directory)

    # We delete pycache
    if os.path.exists('./__pycache__'):
        shutil.rmtree('./__pycache__')

    print('\n', 'DONE!', '\n')
    print('Execution Time: ', time.time() - start, "\n")








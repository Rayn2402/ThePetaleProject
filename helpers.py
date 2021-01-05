import os
import csv


def colsForSql(cols):
    """
    Transform a list of strings containing the name of columns to a string ready to use in a SQL query, Ex: ["name","age","gender"]  ==> "name","age","gender"

    :param cols: the list of column names
    :return: a string

    """
    cols = list(map(lambda c: '"'+c+'"', cols))

    return ",".join(cols)


def writeCsvFile(data, filename):
    """
    Function that takes a list of python dictionaries and generate a CSV file from them

    :param data: the list of dictionaries
    :param filename: the of the csv file that will be generated
    :generate a csv file

    The generated file will be in the folder missing_data

    """
    try:
        # we check if the folder exists
        if not os.path.exists('missing_data'):
            # if the folder wasn't created, we create it
            os.makedirs('missing_data')
        # we open a new file
        with open(f"./missing_data/{filename}", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
            # we write the headers of the CSV file
            writer.writeheader()
            # we write all the data
            for item in data:
                writer.writerow(item)
    except IOError:
        print("I/O error")


def timeDeltaToMonths(timeDelta):
    """
    Function that transform from the type TimeDelta to months

    :param data: timeDelta object
    :return: number of month
    """

    return round(timeDelta.total_seconds() / 2592000, 2)


def extract_var_id(var_name):
    """
    Function that return the id of the varible of a given variable

    :param var_name: the variable name
    :return:a string
    """

    return var_name.split()[0]


def check_categorical_var(data):
    """
    Function that get the data of a variable and return True if this variable is categorical

    :param data:the data of the variable
    :return: Bool
    """
    values_are_string = False
    for item in data:
        if(item != None):
            if(isinstance(item, str)):
                values_are_string = True
                return True

    if(len(data.unique()) > 10):
        return False
    return True

"""
Filename: get_obesity_classification_metrics.py

Author: Nicolas Raymond

Description: Adds the sensitivity, specificity and bAcc scores to obesity experiments
             at a given path.
"""
import sys
from argparse import ArgumentParser
from os.path import dirname, realpath
from pandas import DataFrame


def argument_parser():
    """
    This function defines a parser with the options to run the script
    """
    # Create a parser
    parser = ArgumentParser(usage='\n python get_classification_metrics.py -p [path] -load_target',
                            description="Calculate the classification and add them to the experiments'"
                                        " records.")

    parser.add_argument('-p', '--path', type=str,
                        help="Path where experiment directories are stored")

    parser.add_argument('-lt', '--load_target', default=False, action='store_true',
                        help='If true, targets are load from a database instead of being calculated')

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))
    from src.data.extraction.constants import AGE, CHILDREN_OBESITY_PERCENTILE, OBESITY, OBESITY_TARGET, SEX
    from src.data.extraction.data_management import PetaleDataManager
    from src.utils.results_analyses import get_classification_metrics

    # Class generation function
    def get_class_labels(df: DataFrame, input_column: str, new_column: str) -> DataFrame:
        """
        Generates class labels from real valued predictions

        Args:
            df: pandas dataframe with predictions
            input_column: name of the column with the values from which we infer the classes
            new_column: name of the column that will store the classes

        Returns: pandas dataframe

        """
        # We had a new column filled with zeros
        df[new_column] = 0

        # We switch 0 for 1 according to the classification criteria
        df.loc[(df[SEX] == 'Women') & (df[AGE] >= 18) & (df[input_column] > 35), [new_column]] = 1
        df.loc[(df[SEX] == 'Men') & (df[AGE] >= 18) & (df[input_column] > 25), [new_column]] = 1

        for sex, val in CHILDREN_OBESITY_PERCENTILE.items():
            for age, percentile in val.items():
                filter_ = (df[SEX] == sex) & (df[AGE] >= age) & (df[AGE] < age + 0.5) & (df[input_column] >= percentile)
                df.loc[filter_, [new_column]] = 1

        return df

    # Arguments parsing
    args = argument_parser()

    # Calculation of metrics
    m = PetaleDataManager() if args.load_target else None
    get_classification_metrics(data_manager=m,
                               target_table=OBESITY_TARGET,
                               target_column=OBESITY,
                               experiments_path=args.path,
                               class_generator_function=get_class_labels)

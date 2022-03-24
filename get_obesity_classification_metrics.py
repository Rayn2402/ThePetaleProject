"""
Filename: get_obesity_classification_metrics.py

Author: Nicolas Raymond

Description: Adds the sensitivity, specificity and bAcc scores to obesity experiments
             at a given path.
"""
from pandas import DataFrame
from src.data.extraction.constants import AGE, OBESITY, OBESITY_PERCENTILE, OBESITY_TARGET, SEX, TOTAL_BODY_FAT
from src.utils.results_analysis import CLASS_PRED, get_classification_metrics, REG_PRED

if __name__ == '__main__':

    # Class generation function
    def get_class_labels(df: DataFrame) -> DataFrame:
        """
        Generates class labels from real valued predictions

        Args:
            df: df with predictions

        Returns: df

        """
        df.loc[(df[SEX] == 'Women') & (df[AGE] > 18) & (df[REG_PRED] > 35), [CLASS_PRED]] = 1
        df.loc[(df[SEX] == 'Men') & (df[AGE] > 18) & (df[REG_PRED] > 25), [CLASS_PRED]] = 1
        df.loc[(df[AGE] < 18) & (df[REG_PRED] >= OBESITY_PERCENTILE), [CLASS_PRED]] = 1

        return df


    get_classification_metrics(target_table_name=OBESITY_TARGET,
                               target_column_name=OBESITY,
                               class_generator_function=get_class_labels)

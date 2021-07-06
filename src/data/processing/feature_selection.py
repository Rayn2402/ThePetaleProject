"""
Authors : Nicolas Raymond

File that stores feature selector object, that removes unimportant features
"""
from os.path import join
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from src.data.processing.datasets import CustomDataset
from typing import Tuple, List, Optional


class FeatureSelector:
    """
    Object in charge of selecting the most important features of the dataset
    """

    def __init__(self, importance_threshold: float, records_path: Optional[str] = None):
        """
        Sets protected attributes

        Args:
            importance_threshold: cumulative importance of features selected
            records_path: paths used to store figures and importance table
        """
        self.__importance_thresh = importance_threshold
        self.__records_path = records_path

    def __call__(self, dataset: CustomDataset):
        """
        Extracts most important features using a random forest

        Args:
            dataset: custom dataset

        Returns: List of cont_cols preserved, List of cat_cols preserved
        """
        # Extract feature importance
        fi_table = self.get_features_importance(dataset)

        # Select the subset of selected feature
        selected_features = fi_table.loc[fi_table['status'] == 'selected', 'features'].values

        # Save selected cont_cols and cat_cols
        cont_cols = [c for c in dataset.cont_cols if c in selected_features]
        cat_cols = [c for c in dataset.cat_cols if c in selected_features]

        # Save records in a csv
        if self.__records_path is not None:
            fi_table.to_csv(join(self.__records_path, "feature_selection_records.csv"), index=False)

        return cont_cols, cat_cols

    def get_features_importance(self, dataset: CustomDataset) -> DataFrame:
        """
        Trains a random forest (with default sklearn hyperparameters) to solve the classification
        or regression problems and uses it to extract feature importance.

        Args:
            dataset: custom dataset

        Returns: Dataframe with feature importance
        """
        # Extraction of current training mask
        mask = dataset.train_mask

        # Selection of model
        if dataset.classification:
            model = RandomForestClassifier(n_jobs=-1, oob_score=True).fit(dataset.x.iloc[mask], dataset.y[mask])
        else:
            model = RandomForestRegressor(n_jobs=-1, oob_score=True).fit(dataset.x.iloc[mask], dataset.y[mask])

        # Creation of feature importance table
        fi_table = DataFrame({'features': dataset.x.columns, 'imp': model.feature_importances_}
                             ).sort_values('imp', ascending=False)

        # Addition of a column that indicates if the feature is selected
        cumulative_imp = 0
        status_list = []
        for index, row in fi_table.iterrows():
            cumulative_imp += row['imp']
            status_list.append('selected')
            if cumulative_imp > self.__importance_thresh:
                break

        status_list += ['rejected']*(fi_table.shape[0] - len(status_list))
        fi_table['status'] = status_list

        # Rounding of importance values
        fi_table['imp'] = fi_table['imp'].apply(lambda x: round(x, 4))

        return fi_table

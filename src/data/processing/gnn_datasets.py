"""
Filename: gnn_datasets.py

Author: Nicolas Raymond

Description: File used to store datasets associated to GNNs

Date of last modification: 2022/05/12
"""

from dgl import DGLGraph, graph
from pandas import DataFrame
from src.data.processing.datasets import MaskType, PetaleDataset
from src.data.processing.feature_selection import FeatureSelector
from torch import cat, diag, eye, mm, ones, sqrt, tensor, topk, zeros
from typing import Any, Dict, List, Optional, Tuple


class PetaleKGNNDataset(PetaleDataset):
    """
    Dataset used to train, valid and test Graph Neural Network.
    K-GNN means that the graph structure is built using the K-nearest neighbors
    on the specified columns.
    """
    EUCLIDEAN: str = 'euclidean'
    COSINE: str = 'cosine'

    def __init__(self,
                 df: DataFrame,
                 target: str,
                 k: int = 5,
                 self_loop: bool = True,
                 weighted_similarity: bool = False,
                 cont_cols: Optional[List[str]] = None,
                 cat_cols: Optional[List[str]] = None,
                 gene_cols: Optional[List[str]] = None,
                 feature_selection_groups: Optional[List[List[str]]] = None,
                 conditional_cat_col: Optional[str] = None,
                 classification: bool = True):
        """
        Sets protected and public attributes of our custom dataset class

        Args:
            df: dataframe with the original data
            target: name of the column with the targets
            k: number of closest neighbors used to build the population graph
            self_loop: if True, self loop will be added to nodes in the graph
            weighted_similarity: if True, the weights will assigned to features
                                 during similarities calculation
            cont_cols: list of column names associated with continuous data
            cat_cols: list of column names associated with categorical data
            gene_cols: list of categorical column names that must be considered as genes
            feature_selection_groups: list with list of column names to consider together
                                      in group-wise feature selection
            conditional_cat_col: name of column for which items need to share the same
                                 value in order to be allowed to be connected in the population graph
            classification: true for classification task, False for regression

        """
        # We save the conditional categorical column given
        if conditional_cat_col is not None:
            if conditional_cat_col not in cat_cols:
                raise ValueError(f'{conditional_cat_col} not found among cat_cols')
        self._conditional_cat_col = conditional_cat_col

        # We set the some attributes to default value
        self._feature_imp_extractor = None
        self._neighbors_count = None
        self._nearest_neighbors_idx = None
        self._similarities = None

        # We set the similarity measure
        if cat_cols is None or (len(cat_cols) == 1 and cat_cols[0] == self._conditional_cat_col):
            self._similarity_measure = PetaleKGNNDataset.EUCLIDEAN
        else:
            self._similarity_measure = PetaleKGNNDataset.COSINE

        if weighted_similarity:
            self._w_sim = True
            self._feature_imp_extractor = FeatureSelector(threshold=[1], cumulative_imp=[True], seed=1010710)
        else:
            self._w_sim = False

        # We save the number of k-nearest neighbors and the self-loop attribute
        self._self_loop = self_loop
        self._k = k

        # We set train, valid and test subgraphs data to default value
        self._subgraphs = {MaskType.TRAIN: tuple(), MaskType.VALID: tuple(), MaskType.TEST: tuple()}

        # We use the _init_ of the parent class
        super().__init__(df=df,
                         target=target,
                         cont_cols=cont_cols,
                         cat_cols=cat_cols,
                         gene_cols=gene_cols,
                         feature_selection_groups=feature_selection_groups,
                         classification=classification,
                         to_tensor=True)

    @property
    def train_subgraph(self) -> Tuple[DGLGraph, Dict[int, int], List[int]]:
        return self._subgraphs[MaskType.TRAIN]

    @property
    def test_subgraph(self) -> Tuple[DGLGraph, Dict[int, int], List[int]]:
        return self._subgraphs[MaskType.TEST]

    @property
    def valid_subgraph(self) -> Tuple[DGLGraph, Dict[int, int], List[int]]:
        return self._subgraphs[MaskType.VALID]

    def _build_graph_from_knn(self,
                              idx: List[int],
                              top_n_neighbors: List[List[int]]) -> Tuple[DGLGraph, Dict[int, int]]:
        """
        Builds a graph by connecting each node i in the idx list to its k-nearest neighbors
        ordered in the top_n_neighbors[i] list.

        Args:
            idx: list of nodes idx to use in order to build the graph
            top_n_neighbors: list of list containing idx of closest neighbors of each node

        Returns: DGLgraph, dictionary mapping each idx to its position the graph
        """
        # We create the dictionary mapping the original idx to their position in the graph
        idx_map = self._create_idx_map(idx)

        # We build the edges of the graph using the original idx
        u, v = [], []  # Node ids
        e = []  # Edges weights
        if not self._self_loop:
            for i in idx:
                nb_neighbor = min(len(top_n_neighbors[i]), self._k)
                neighbor_idx = top_n_neighbors[i][:nb_neighbor]
                u += neighbor_idx
                v += [i] * nb_neighbor
                e += self._similarities[i, neighbor_idx]/self._similarities[i, neighbor_idx].sum()
        else:
            for i in idx:
                nb_neighbor = min(len(top_n_neighbors[i]), self._k)
                neighbor_idx = [i] + top_n_neighbors[i][:nb_neighbor]
                u += neighbor_idx
                v += [i] * (nb_neighbor + 1)
                e += self._similarities[i, neighbor_idx] / self._similarities[i, neighbor_idx].sum()

        # We replace the idx with their node position
        u = [idx_map[n] for n in u]
        v = [idx_map[n] for n in v]
        u, v = tensor(u).long(), tensor(v).long()

        # We build the graph, saves the original ids and the edges weights
        g = graph((u, v))
        g.ndata['IDs'] = tensor(idx)
        g.edata['w'] = tensor(e)

        # We return the graph and the mapping of each idx to its position in the graph
        return g, idx_map

    def _build_neighbors_filter_mat(self) -> tensor:
        """
        Creates an (N,N) tensor where element i,j is a 1 if item-i and item-j
        shares the same value for the conditional_cat_col. Element i,j are zeros otherwise.

        Returns: (N,N) tensor
        """
        # If no conditional column was given
        if self._conditional_cat_col is None:
            return ones(self._n, self._n)

        # We initialize a matrix filled with zeros
        neighbors_filter = zeros(self._n, self._n)

        # We fill the upper triangle of the edges matrix
        df = self.get_imputed_dataframe()
        for value in self.encodings[self._conditional_cat_col].values():

            # Idx of patients sharing same categorical value
            idx_subset = df.loc[df[self._conditional_cat_col] == value].index.to_numpy()

            # For patient with common value we add 1
            k = 0
            for i in idx_subset:
                for j in idx_subset[k+1:]:
                    neighbors_filter[i, j] += 1
                k += 1

        # We add it to its transpose to have the complete matrix
        neighbors_filter = neighbors_filter + neighbors_filter.t()

        return neighbors_filter

    def _build_pop_subgraph(self,
                            native_node_idx: List[int],
                            added_node_idx: Optional[List[int]] = None
                            ) -> Tuple[DGLGraph, Dict[int, int], List[int]]:
        """
        Builds a graph where all nodes can only be connected to any native node

        Args:
            native_node_idx: list of idx associated to native nodes
            added_node_idx: list of idx associated to added nodes

        Returns: DGLgraph, dictionary mapping each idx to its position the graph, list with all the idx
        """
        # We extract the nearest neighbors idx list
        nn_idx = self._nearest_neighbors_idx.clone().detach().tolist()

        # We filter nodes possible neighbors
        added_node_idx = [] if added_node_idx is None else added_node_idx
        all_nodes_idx = native_node_idx + added_node_idx
        for i in all_nodes_idx:
            nn_idx[i] = nn_idx[i][:self._neighbors_count[i]]
            nn_idx[i] = [j for j in nn_idx[i] if j in native_node_idx]

        # We build the graph and the node mapping
        g, m = self._build_graph_from_knn(all_nodes_idx, nn_idx)

        return g, m, all_nodes_idx

    def compute_cosine_sim(self) -> tensor:
        """
        Calculates cosine similarities between individuals

        Returns: (N, N) tensor with distances
        """
        # We extract data
        x = []
        if len(self._cont_idx) > 0:
            x.append(self.x_cont)
        if len(self._cat_idx) > 0:
            x.append(self.get_one_hot_encodings(cat_cols=self._cat_cols))
        x = cat(x, dim=1)

        # We extract feature importance
        if self._feature_imp_extractor is not None:
            fi = self._get_features_importance()
            fi = fi.reshape(1, -1)
        else:
            fi = ones(x.shape)

        # We compute weighted dot products between individuals
        x_prime = mm(x*fi, x.t())

        # We compute weighted norms
        norms = sqrt((pow(x, 2)*fi).sum(dim=1)).reshape(-1, 1)

        # We return cosine similarities
        return x_prime / mm(norms, norms.t())

    def _compute_euclidean_dist(self) -> tensor:
        """
        Calculates squared euclidean distances between individuals

        Returns: (N, N) tensor with distances
        """
        # We calculate the weight of the continuous features
        if self._feature_imp_extractor is not None:
            fi = self._get_features_importance()
            fi = fi[:len(self.cont_idx)]
            fi /= fi.sum()
            weight_mat = diag(fi)
        else:
            weight_mat = eye(len(self.cont_idx))

        # We compute squared euclidean distances
        numerical_data = self.x_cont
        euclidean_dist = zeros(self._n, self._n, requires_grad=False)
        for i in range(self._n):
            for j in range(i+1, self._n):
                diff = (numerical_data[i, :] - numerical_data[j, :]).reshape(-1, 1)
                euclidean_dist[i, j] = mm(mm(diff.t(), weight_mat), diff)

        # We add the matrix to its transpose to have all distances
        euclidean_dist = euclidean_dist + euclidean_dist.t()

        return euclidean_dist

    def _compute_similarities(self) -> tensor:
        """
        Computes similarities between individuals

        Returns: (N, N) tensor
        """
        if self._similarity_measure == PetaleKGNNDataset.EUCLIDEAN:
            sim = 1 / (self._compute_euclidean_dist() + ones(self._n, self._n))
        else:
            sim = self.compute_cosine_sim()

        return sim

    def _get_features_importance(self) -> tensor:
        """
        Calculates the feature importance of each feature in the dataset
        according to the labels

        Returns: tensor with feature importance
        """
        # We initialize a dictionary to store feature importance
        fi_dict = {}
        for i in range(len(self._cont_idx)):
            fi_dict[self._cont_cols[i]] = 1

        cat_sizes = self.cat_sizes
        for i in range(len(self._cat_idx)):
            fi_dict[self._cat_cols[i]] = cat_sizes[i]

        # We calculate feature importance
        fi_table = self._feature_imp_extractor.get_features_importance(self)

        # We store the feature importance in the dictionary
        for _, row in fi_table.iterrows():
            fi_dict[row['features']] = [row['imp']]*fi_dict[row['features']]

        # We flatten the values of the dictionary
        fi = []
        for v in fi_dict.values():
            fi += v

        # We return a tensor
        return tensor(fi)

    def _set_subgraphs_data(self) -> None:
        """
        Sets subgraphs data after masks update

        Returns: None
        """
        # Set the subgraph associated to the training set and a map matching each training
        # index to its physical position in the train mask
        self._subgraphs[MaskType.TRAIN] = self._build_pop_subgraph(native_node_idx=self.train_mask)

        # Set the subgraph associated to test and a map matching each test
        # index to its physical position in the train + test mask
        self._subgraphs[MaskType.TEST] = self._build_pop_subgraph(native_node_idx=self.train_mask + self.valid_mask,
                                                                  added_node_idx=self.test_mask)
        if len(self.valid_mask) != 0:

            # Set the subgraph associated to validation and a map matching each valid
            # index to its physical position in the train + valid mask
            self._subgraphs[MaskType.VALID] = self._build_pop_subgraph(native_node_idx=self.train_mask,
                                                                       added_node_idx=self.valid_mask)

    def _set_nearest_neighbors(self) -> None:
        """
        Sets, for each item in the dataset, an ordered sequence of idx representing the nearest neighbors.
        Also saves the number of possible neighbors for each item.

        Returns: None
        """
        # We calculate similarities between each item
        similarities = self._compute_similarities()

        # We turn some similarities to zeros if a conditional column was given
        filter_mat = self._build_neighbors_filter_mat()
        self._similarities = similarities * filter_mat

        # We count the number of ones in each row of the filter mat
        self._neighbors_count = (filter_mat == 1).sum(dim=1)

        # We get the idx of the (n-1)-nearest neighbors of each item (excluding themselves)
        _, self._nearest_neighbors_idx = topk(self._similarities, k=(self._n - 1), dim=1)

    def get_arbitrary_subgraph(self, idx: List[int]) -> Tuple[DGLGraph, Dict[int, int], List[int]]:
        """
        Returns a tuple with :
        1 - homogeneous subgraph with only the nodes associated to idx in the list
        2-  dictionary mapping each idx to its position in the list

        Args:
            idx: list of idx such as masks

        Returns: homogeneous graph, dict matching each training index to its physical position in idx list,
                 list with all the idx
        """

        # We remove the that are part of the native node
        added_idx = [i for i in idx if i not in self.train_mask]

        # We build the subgraph
        g, m, idx = self._build_pop_subgraph(native_node_idx=self.train_mask, added_node_idx=added_idx)

        return g, m, idx

    def update_masks(self,
                     train_mask: List[int],
                     test_mask: List[int],
                     valid_mask: Optional[List[int]] = None) -> None:
        """
        First, updates the train, valid and test masks and preprocess the data available
        according to the current statistics of the training data.

        Then, rebuild the graph by running k-nearest neighbors algorithm.

        Second, updates train, valid and test subgraph and idx map.

        """
        # We first update masks as usual for datasets
        PetaleDataset.update_masks(self, train_mask=train_mask, test_mask=test_mask, valid_mask=valid_mask)

        # If we are not calling update_masks for initialization purpose
        if len(test_mask) != 0:

            # We set the attributes related to nearest neighbors
            self._set_nearest_neighbors()

            # We update the subgraphs data
            self._set_subgraphs_data()

    def create_subset(self,
                      cont_cols: Optional[List[str]] = None,
                      cat_cols: List[str] = None) -> Any:
        """
        Returns a subset of the current dataset using the given cont_cols and cat_cols

        Args:
            cont_cols: list of continuous columns
            cat_cols: list of categorical columns

        Returns: instance of the PetaleKGNNDataset class
        """
        subset = self._retrieve_subset_from_original(cont_cols, cat_cols)
        gene_cols = None if len(self._gene_cols) == 0 else [c for c in self._gene_cols if c in cat_cols]
        return PetaleKGNNDataset(df=subset,
                                 k=self._k,
                                 self_loop=self._self_loop,
                                 weighted_similarity=self._w_sim,
                                 target=self.target,
                                 cont_cols=cont_cols,
                                 cat_cols=cat_cols,
                                 gene_cols=gene_cols,
                                 conditional_cat_col=self._conditional_cat_col,
                                 classification=self.classification)

    def create_superset(self,
                        data: DataFrame,
                        categorical: bool = False,
                        gene: bool = False) -> Any:
        """
        Returns a superset of the current dataset by including the given data

        Args:
            data: pandas dataframe with 2 columns
                  First column must be PARTICIPANT ids
                  Second column must be the feature we want to add
            categorical: True if the new feature is categorical
            gene: True if the new feature is considered as a gene

        Returns: instance of the PetaleKGNNDataset class
        """
        # We build the augmented dataframe
        df, cont_cols, cat_cols, gene_cols = self._get_augmented_dataframe(data, categorical, gene)

        return PetaleKGNNDataset(df=df,
                                 target=self.target,
                                 k=self._k,
                                 self_loop=self._self_loop,
                                 weighted_similarity=self._w_sim,
                                 cont_cols=cont_cols,
                                 cat_cols=cat_cols,
                                 gene_cols=gene_cols,
                                 conditional_cat_col=self._conditional_cat_col,
                                 classification=self.classification)

    @staticmethod
    def _create_idx_map(idx: List[int]) -> Dict[int, int]:
        """
        Creates a dictionary mapping each element of a list to its position in the list

        Args:
            idx: list of idx

        Returns: dictionary
        """
        return {v: i for i, v in enumerate(idx)}

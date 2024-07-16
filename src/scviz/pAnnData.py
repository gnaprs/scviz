import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc

import copy

from scipy import sparse
from scipy.stats import variation
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer

import umap.umap_ as umap

import seaborn as sns
import matplotlib.pyplot as plt

from pandas.testing import assert_frame_equal

from scviz import utils

from typing import (  # Meta  # Generic ABCs  # Generic
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    List
)

# TODO!: methods to write
# easy way to assign obs, names, var to prot and pep
# def set_prot_obs(self, obs):

class pAnnData:
    """
    Class for storing protein and peptide data, with additional relational data between protein and peptide.
    
    Parameters
    ----------
    prot : np.ndarray | sparse.spmatrix
        Protein data matrix.
    pep : np.ndarray | sparse.spmatrix
        Peptide data matrix.
    rs : np.ndarray | sparse.spmatrix
        Protein x peptide relational data. Only if both protein and peptide data are provided.
    history : List[str]
        List of actions taken on the data.
    summary : pd.DataFrame
        Summary of the data, typically used for filtering.
    """

    def __init__(self, 
                 prot = None, # np.ndarray | sparse.spmatrix 
                 pep = None, # np.ndarray | sparse.spmatrix
                 rs = None): # np.ndarray | sparse.spmatrix, protein x peptide relational data

        if prot is not None:
            self._prot = ad.AnnData(prot)
        else:
            self._prot = None
        if pep is not None:
            self._pep = ad.AnnData(pep)
        else:
            self._pep = None
        if rs is not None:
            self._set_RS(rs)
        else:
            self._rs = None

        self._history = []
        self._summary = pd.DataFrame()
    
    # GETTERS
    @property
    def prot(self):
        return self._prot

    @property
    def pep(self):
        return self._pep
    
    @property
    def rs(self):
        return self._rs

    @property
    def history(self):
        return self._history

    @property
    def summary(self):
        return self._summary

    # SETTERS
    @prot.setter
    def prot(self, value: ad.AnnData):
        self._prot = value

    @pep.setter
    def pep(self, value: ad.AnnData):
        self._pep = value

    @rs.setter
    def rs(self, value):
        self._set_RS(value)

    @history.setter
    def history(self, value):
        self._history = value

    @summary.setter
    def summary(self, value: pd.DataFrame):
        self._summary = value
        self._update_obs()

    # UTILITY FUNCTIONS
    def _set_RS(self, rs):
        # print rs shape, as well as protein and peptide shape if available
        print(f"Setting rs matrix with dimensions {rs.shape}")
        # assert that the dimensions of rs match either protein x peptide or peptide x protein
        assert ((self.prot is None or rs.shape[0] == self.prot.shape[1]) and (self.pep is None or rs.shape[1] == self.pep.shape[1])) or \
            ((self.prot is None or rs.shape[1] == self.prot.shape[1]) and (self.pep is None or rs.shape[0] == self.pep.shape[1])), \
            f"The dimensions of rs ({rs.shape}) must match either protein x peptide ({self.prot.shape[1] if self.prot is not None else None} x {self.pep.shape[1] if self.pep is not None else None}) or peptide x protein ({self.pep.shape[1] if self.pep is not None else None} x {self.prot.shape[1] if self.prot is not None else None})"

        # check dimensions of rs, make sure protein (row) x peptide (column) format
        if self.prot is not None and rs.shape[0] != self.prot.shape[1]:
            print("Transposing rs matrix to protein x peptide format")
            rs = rs.T
        self._rs = sparse.csr_matrix(rs)

    def __repr__(self):
        if self.prot is not None:
            prot_shape = f"{self.prot.shape[0]} files by {self.prot.shape[1]} proteins"
            prot_obs = ', '.join(self.prot.obs.columns[:5]) + ('...' if len(self.prot.obs.columns) > 5 else '')
            prot_var = ', '.join(self.prot.var.columns[:5]) + ('...' if len(self.prot.var.columns) > 5 else '')
            prot_obsm = ', '.join(self.prot.obsm.keys()) + ('...' if len(self.prot.obsm.keys()) > 5 else '')
            prot_layers = ', '.join(self.prot.layers.keys()) + ('...' if len(self.prot.layers.keys()) > 5 else '')
            prot_info = f"Protein (shape: {prot_shape})\nobs: {prot_obs}\nvar: {prot_var}\nobsm: {prot_obsm}\nlayers: {prot_layers}"
        else:
            prot_info = "Protein: None"

        if self.pep is not None:
            pep_shape = f"{self.pep.shape[0]} files by {self.pep.shape[1]} peptides"
            pep_obs = ', '.join(self.pep.obs.columns[:5]) + ('...' if len(self.pep.obs.columns) > 5 else '')
            pep_var = ', '.join(self.pep.var.columns[:5]) + ('...' if len(self.pep.var.columns) > 5 else '')
            pep_layers = ', '.join(self.pep.layers.keys()) + ('...' if len(self.pep.layers.keys()) > 5 else '')
            pep_info = f"Peptide (shape: {pep_shape})\nobs: {pep_obs}\nvar: {pep_var}\nlayers: {pep_layers}"
        else:
            pep_info = "Peptide: None"

        if self._rs is not None:
            rs_shape = f"{self._rs.shape[0]} proteins by {self._rs.shape[1]} peptides"
            rs_info = f"RS (shape: {rs_shape})\n"
        else:
            rs_info = "RS: None"

        return (f"pAnnData object\n"
                f"{prot_info}\n\n"
                f"{pep_info}\n\n"
                f"{rs_info}\n")
    
    def _has_data(self):
        return self.prot is not None or self.pep is not None

    def _check_data(self, on):
        if on not in ['protein', 'peptide']:
            raise ValueError("Invalid input: on must be either 'protein' or 'peptide'.")
        elif on == 'protein' and self.prot is None:
            raise ValueError("No protein data found in AnnData object.")
        elif on == 'peptide' and self.pep is None:
            raise ValueError("No peptide data found in AnnData object.")
        else:
            return True

    def _update_summary(self):
        if self.prot is not None:
            self.prot.obs['protein_quant'] = np.sum(~np.isnan(self.prot.X.toarray()), axis=1) / self.prot.X.shape[1]
            self.prot.obs['protein_count'] = np.sum(~np.isnan(self.prot.X.toarray()), axis=1)
            if 'X_mbr' in self.prot.layers:
                self.prot.obs['mbr_count'] = (self.prot.layers['X_mbr'] == 'Peak Found').sum(axis=1)
                self.prot.obs['high_count'] = (self.prot.layers['X_mbr'] == 'High').sum(axis=1)

        if self.pep is not None:
            self.pep.obs['peptide_quant'] = np.sum(~np.isnan(self.pep.X.toarray()), axis=1) / self.pep.X.shape[1]
            self.pep.obs['peptide_count'] = np.sum(~np.isnan(self.pep.X.toarray()), axis=1)
            if 'X_mbr' in self.pep.layers:
                self.pep.obs['mbr_count'] = (self.pep.layers['X_mbr'] == 'Peak Found').sum(axis=1)
                self.pep.obs['high_count'] = (self.pep.layers['X_mbr'] == 'High').sum(axis=1)
        
        if self.prot is not None:
            self._summary = self.prot.obs.copy()
            if self.pep is not None:
                for col in self.pep.obs.columns:
                    if col not in self._summary.columns:
                        self._summary[col] = self.pep.obs[col]
        else:
            self._summary = self.pep.obs.copy()

        self._previous_summary = self._summary.copy()
        
    def _update_obs(self):
        # function to update obs with summary data (if user edited summary data)
        if not self._has_data():
            return

        def update_obs_with_summary(obs, summary, ignore_keyword):
            ignored_columns = []
            for col in summary.columns:
                if ignore_keyword in col:
                    ignored_columns.append(col)
                    continue
                obs[col] = summary[col]
            return ignored_columns
            

        if self.prot is not None:
            if not self.prot.obs.index.equals(self._summary.index):
                raise ValueError("Index of summary does not match index of prot.obs")
            ignored_columns_prot = update_obs_with_summary(self.prot.obs, self._summary, "pep")
        else:
            ignored_columns_prot = None
        if self.pep is not None:
            if not self.pep.obs.index.equals(self._summary.index):
                raise ValueError("Index of summary does not match index of pep.obs")
            ignored_columns_pep = update_obs_with_summary(self.pep.obs, self._summary, "prot")
        else:
            ignored_columns_pep = None

        history_statement = "Updated obs with summary data. "
        if ignored_columns_prot:
            history_statement += f"Ignored columns in prot.obs: {', '.join(ignored_columns_prot)}. "
        if ignored_columns_pep:
            history_statement += f"Ignored columns in pep.obs: {', '.join(ignored_columns_pep)}. "
        self._history.append(history_statement)

    def _append_history(self, action):
        self._history.append(action)

    def _check_rankcol(self, on = 'protein', class_values = None):
        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        if class_values is None:
            raise ValueError("class_values must be None")

        for class_value in class_values:
            average_col = f'Average: {class_value}'
            rank_col = f'Rank: {class_value}'
            if average_col not in adata.var.columns or rank_col not in adata.var.columns:
                raise ValueError(f"Class name not found in .var. Please run plot_rankquank() beforehand and check that the input matches the class names in {on}.var['Average: ']")


    def print_history(self):
        formatted_history = "\n".join(f"{i}: {action}" for i, action in enumerate(self._history, 1))
        print("-------------------------------\nHistory:\n-------------------------------\n"+formatted_history)

    # -----------------------------
    # EDITING FUNCTIONS
    def copy(self):
        """
        Returns a deep copy of the pAnnData object.
        """
        return copy.deepcopy(self)

    def set_X(self, layer, on = 'protein'):
        # defines which layer to set X to
            if not self._check_data(on):
                pass

            if on == 'protein':
                if layer not in self.prot.layers:
                    raise ValueError(f"Layer {layer} not found in protein data.")
                self.prot.X = self.prot.layers[layer]
                print(f"Set {on} data to layer {layer}.")

            else:
                if layer not in self.pep.layers:
                    raise ValueError(f"Layer {layer} not found in peptide data.")
                self.pep.X = self.pep.layers[layer]
                print(f"Set {on} data to layer {layer}.")

            self._history.append(f"{on}: Set X to layer {layer}.")

    def filter(self, condition = None, return_copy = True, file_list = None):
        # docstring
        # example usage: pdata.filter("protein_count > 1000")
        if not self._has_data():
            pass

        if self._summary is None:
            self._update_summary()

        if return_copy:
            pdata = self.copy()

            if condition is not None:
                filtered_queries = pdata._summary.query(condition)

                if pdata.prot is not None:
                    pdata.prot = pdata.prot[filtered_queries.index]
                
                if pdata.pep is not None:
                    pdata.pep = pdata.pep[filtered_queries.index]

                print(f"Returning a copy of filtered data based on condition: {condition}. Number of samples dropped: {len(pdata._summary) - len(filtered_queries)}.")
                pdata._append_history(f"Filtered data based on condition: {condition}")
                pdata._update_summary()

                return pdata
            elif file_list is not None:
                if pdata.prot is not None:
                    pdata.prot = pdata.prot[pdata.prot.obs_names.isin(file_list)]
                
                if pdata.pep is not None:
                    pdata.pep = pdata.pep[pdata.pep.obs_names.isin(file_list)]

                print(f"Returning a copy of filtered data based on file list. Number of samples dropped: {len(pdata._summary) - len(file_list)}.")
                pdata._append_history(f"Filtered data based on file list.")
                pdata._update_summary()

                return pdata
        else:
            if condition is not None:
                filtered_queries = self._summary.query(condition)

                if self.prot is not None:
                    self.prot = self.prot[filtered_queries.index]

                if self.pep is not None:
                    self.pep = self.pep[filtered_queries.index]

                print(f"Filtered and modified data based on condition: {condition}. Number of samples dropped: {len(self._summary) - len(filtered_queries)}.")
                self._append_history(f"Filtered data based on condition: {condition}")
                self._update_summary()
            elif file_list is not None:
                if self.prot is not None:
                    self.prot = self.prot[self.prot.obs_names.isin(file_list)]
                
                if self.pep is not None:
                    self.pep = self.pep[self.pep.obs_names.isin(file_list)]

                print(f"Filtered and modified data based on file list. Number of samples dropped: {len(self._summary) - len(file_list)}.")
                self._append_history(f"Filtered data based on file list.")
                self._update_summary()

    # -----------------------------
    # PROCESSING FUNCTIONS
    def cv(self, layer = "X_raw", classes = None, on = 'protein'):
        if not self._check_data(on):
            pass

        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        if layer == "X":
            # do nothing
            pass
        elif layer in adata.layers.keys():
            self.set_X(layer = layer, on = on)

        if classes is None:
            # combine all .obs columns per row into one string
            quant_col_index = adata.obs.columns.get_loc(next(col for col in adata.obs.columns if "_quant" in col))
            selected_columns = adata.obs.iloc[:, :quant_col_index]
            classes_list = selected_columns.apply(lambda x: '_'.join(x), axis=1).unique()
            classes = selected_columns.columns.tolist()
        elif isinstance(classes, str):
            # check if classes is one of the columns of adata.obs
            if classes not in adata.obs.columns:
                raise ValueError(f"Invalid value for 'classes'. '{classes}' is not a column in adata.obs.")
            
            classes_list = adata.obs[classes].unique()
        elif isinstance(classes, list):
            # check if all classes are columns of adata.obs
            if not all([c in adata.obs.columns for c in classes]):
                raise ValueError(f"Invalid value for 'classes'. Not all elements in '{classes}' are columns in adata.obs.")
            classes_list = adata.obs[classes].apply(lambda x: '_'.join(x), axis=1).unique()
        else:
            raise ValueError("Invalid value for 'classes'. Must be None, a string or a list of strings.")

        for j, class_value in enumerate(classes_list):
            if classes is None:
                values = class_value.split('_')
                print(f'Classes: {classes}, Values: {values}')
                data_filtered = utils.filter(adata, classes, values, suppress_warnings=True)
            elif isinstance(classes, str):
                print(f'Class: {classes}, Value: {class_value}')
                data_filtered = utils.filter(adata, classes, class_value, suppress_warnings=True)
            elif isinstance(classes, list):
                values = class_value.split('_')
                print(f'Classes: {classes}, Values: {values}')
                data_filtered = utils.filter(adata, classes, values, suppress_warnings=True)

        adata.var['CV: '+ class_value] = variation(data_filtered.X.toarray(), axis=0)

    # TODO: ADD THIS
    def rank(self, layer = "X_raw", on = 'protein', ascending = False):
        self._history.append(f"{on}: Ranked {layer} data. Ranking stored in var['rank'].")

    # TODO: add ability to impute within class (provide variable(s) for grouping, etc.) [See ColumnTransformer in sklearn]
    # TODO: add imputation for minimum [https://github.com/scikit-learn/scikit-learn/issues/19783]
    # https://towardsdatascience.com/improve-your-data-preprocessing-with-columntransformer-and-pipelines-b6ff7edd2f77
    def impute(self, layer = "X_raw", method = 'median', on = 'protein', **kwargs):
        # uses scikit-learn imputers
        if not self._check_data(on):
            pass

        # default imputer settings
        missing_values = kwargs.pop('missing_values', np.nan)
        n_neighbors = kwargs.pop('n_neighbors', 2)
        weights = kwargs.pop('weights', "uniform")

        imputers = {
            'median': lambda: SimpleImputer(missing_values=missing_values, strategy='median', keep_empty_features = True, **kwargs),
            'mean': lambda: SimpleImputer(missing_values=missing_values, strategy='mean', keep_empty_features = True, **kwargs),
            'knn': lambda: KNNImputer(n_neighbors=n_neighbors, weights=weights, keep_empty_features = True, **kwargs)
        }


        if method in imputers:
            imputer = imputers[method]()
            layer_name = 'X_impute_' + method

            if on == 'protein':
                self._impute_helper(self.prot, layer, method, layer_name, imputer)
            else:
                self._impute_helper(self.prot, layer, method, layer_name, imputer)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self._history.append(f"{on}: Imputed {layer} data using {method}. Imputed data stored in `{layer_name}`.")

    def _impute_helper(self, data, layer, method, layer_name, imputer):
        '''Helper function for impute, imputes data and stores it in the AnnData object.

        Args:
            data (AnnData): AnnData object to impute.
            layer (str): Layer to impute.
            method (str): Imputation method.
            layer_name (str): Name of the new layer to store imputed data.
            imputer (object): Imputer object.
        '''
        if layer != "X" and layer not in data.layers:
            raise ValueError(f"Layer {layer} not found in data.")

        if method == 'knn':
            if layer == "X":
                data.layers[layer_name] = sparse.csr_matrix(imputer.fit_transform(data.X.toarray()))
            else:
                data.layers[layer_name] = sparse.csr_matrix(imputer.fit_transform(data.layers[layer].toarray()))
        else:
            if layer == "X":
                data.layers[layer_name] = imputer.fit_transform(data.X)
            else:
                data.layers[layer_name] = imputer.fit_transform(data.layers[layer])

        print(f"Imputed data using {method}. New data stored in `{layer_name}`.")

    def neighbor(self, on = 'protein', layer = "X", **kwargs):
        # uses sc.pp.neighbors
        if not self._check_data(on):
            pass
        
        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        if layer == "X":
            # do nothing
            pass
        elif layer in adata.layers.keys():
            self.set_X(layer = layer, on = on)

        sc.pp.neighbors(adata, **kwargs)

        self._append_history(f'{on}: Neighbors fitted on {layer}, stored in obs["distances"] and obs["connectivities"]')
        print(f'{on}: Neighbors fitted on {layer} and and stored in obs["distances"] and obs["connectivities"]')

    def umap(self, on = 'protein', layer = "X", **kwargs):
        # uses sc.tl.umap
        if not self._check_data(on):
            pass
       
        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        # check if neighbor has been run before, look for distances and connectivities in obsp
        if 'neighbors' not in adata.uns:
            print("Neighbors not found in AnnData object. Running neighbors with default settings.")
            self.neighbor(on = on, layer = layer)

        if layer == "X":
            # do nothing
            pass
        elif layer in adata.layers.keys():
            self.set_X(layer = layer, on = on)

        sc.tl.umap(adata, **kwargs)

        self._append_history(f'{on}: UMAP fitted on {layer}, stored in obsm["X_umap"] and uns["umap"]')
        print(f'{on}: UMAP fitted on {layer} and and stored in layers["X_umap"] and uns["umap"]')

    def pca(self, on = 'protein', layer = "X", **kwargs):
        # uses sc.tl.pca
        if not self._check_data(on):
            pass
        
        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        # make sample array
        if layer == "X":
            X = adata.X.toarray()
        elif layer in adata.layers.keys():
            X = adata.layers[layer].toarray()

        print(f'BEFORE: Number of samples|Number of proteins: {X.shape}')
        Xnorm = (X - X.mean(axis=0)) / X.std(axis=0)
        nan_cols = np.isnan(Xnorm).any(axis=0)
        Xnorm = Xnorm[:, ~nan_cols]
        print(f'AFTER: Number of samples|Number of proteins: {Xnorm.shape}')

        pca_data = sc.tl.pca(Xnorm, return_info=True, **kwargs)
        
        adata.obsm['X_pca'] = pca_data[0]
        PCs = np.zeros((pca_data[1].shape[0], nan_cols.shape[0]))
        
        # fill back the 0s where column was NaN in the original data, and thus not used in PCA
        counter = 0
        for i in range(PCs.shape[1]):
            if not nan_cols[i]:
                PCs[:, i] = pca_data[1][:, counter]
                counter += 1

        adata.uns['pca'] = {'PCs': PCs, 'variance_ratio': pca_data[2], 'variance': pca_data[3]}
        
        self._append_history(f'{on}: PCA fitted on {layer}, stored in obsm["X_pca"] and varm["PCs"]')
        print(f'{on}: PCA fitted on {layer} and and stored in layers["X_pca"] and uns["pca"]')

    # TODO: add ability to normalize within class (provide variable(s) for grouping, etc.), median normalization options
    def normalize(self, method = 'scale', on = 'protein', set_X = True, **kwargs):  
        if not self._check_data(on):
            pass

        EPSILON = 1e-10  # small constant

        # median
        # data = data divide by data_median * data.median.median 

        if method == 'scale':
            if on == 'protein':
                row_sums = np.nansum(self.prot.X.toarray(), axis=1)
                max_row_sum = np.max(row_sums)
                self.prot.layers['X_scale'] = sparse.csr_matrix(self.prot.X.toarray() / (row_sums[:, np.newaxis] + EPSILON) * max_row_sum)
                print(f"Normalized {on} data using {method}.")
            else:
                row_sums = np.nansum(self.pep.X.toarray(), axis=1)
                max_row_sum = np.max(row_sums)
                self.pep.layers['X_scale'] = sparse.csr_matrix(self.pep.X.toarray() / (row_sums[:, np.newaxis] + EPSILON) * max_row_sum)
                print(f"Normalized {on} data using {method}.")
        elif method == "l2":
            if on == 'protein':
                self.prot.layers['X_l2'] = normalize(self.prot.X, norm='l2')
                print(f"Normalized {on} data using {method}.")
            else:
                self.pep.layers['X_l2'] = normalize(self.pep.X, norm='l2')
                print(f"Normalized {on} data using {method}.")
        elif method == 'log2':
            if on == 'protein':
                self.prot.layers['X_log2'] = sparse.csr_matrix(np.log2(self.prot.layers['X_raw'].toarray() + 1))
                print(f"Normalized {on} data using {method}.")
            else:
                self.pep.layers['X_log2'] = sparse.csr_matrix(np.log2(self.pep.layers['X_raw'].toarray() + 1))
                print(f"Normalized {on} data using {method}.")
        else:
            raise ValueError(f"Unknown method: {method}")
                  
        self._history.append(f"{on}: Normalized X_raw data using {method} and stored as layers[X_{method}].")
        if set_X:
            layer_name = 'X_' + method
            self.set_X(layer = layer_name, on = on)


def import_proteomeDiscoverer(prot_file: Optional[str] = None, pep_file: Optional[str] = None, obs_columns: Optional[List[str]] = None):
    if not prot_file and not pep_file:
        raise ValueError("At least one of prot_file or pep_file must be provided")
    print("--------------------------\nStarting import...\n--------------------------")
    
    if prot_file:
        # -----------------------------
        print(f"Importing from {prot_file}")
        # PROTEIN DATA
        # check file format, if '.txt' then use read_csv, if '.xlsx' then use read_excel
        if prot_file.endswith('.txt') or prot_file.endswith('.tsv'):
            prot_all = pd.read_csv(prot_file, sep='\t')
        elif prot_file.endswith('.xlsx'):
            print("WARNING: The read_excel function is slower compared to reading .tsv or .txt files. For improved performance, consider converting your data to .tsv or .txt format.")
            prot_all = pd.read_excel(prot_file)
        # prot_X: sparse data matrix
        prot_X = sparse.csr_matrix(prot_all.filter(regex='Abundance: F', axis=1).values).transpose()
        # prot_layers['mbr']: protein MBR identification
        prot_layers_mbr = prot_all.filter(regex='Found in Sample', axis=1).values.transpose()
        # prot_var_names: protein names
        prot_var_names = prot_all['Accession'].values
        # prot_var: protein metadata
        prot_var = prot_all.loc[:, 'Protein FDR Confidence: Combined':'# Razor Peptides']
        # prot_obs_names: file names
        prot_obs_names = prot_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: (F\d+):')[0].values
        # prot_obs: sample typing from the column name
        prot_obs = prot_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: F\d+: (.+)$')[0].values
        prot_obs = pd.DataFrame(prot_obs, columns=['metadata'])['metadata'].str.split(',', expand=True).map(str.strip).astype('category')

        print(f"Number of files: {len(prot_obs_names)}")
        print(f"Number of proteins: {len(prot_var)}")
    else:
        prot_X = prot_layers_mbr = prot_var_names = prot_var = prot_obs_names = prot_obs = None

    if pep_file:
        # -----------------------------
        print(f"Importing from {pep_file}")
        # PEPTIDE DATA
        if pep_file.endswith('.txt') or pep_file.endswith('.tsv'):
            pep_all = pd.read_csv(pep_file, sep='\t')
        elif pep_file.endswith('.xlsx'):
            print("WARNING: The read_excel function is slower compared to reading .tsv or .txt files. For improved performance, consider converting your data to .tsv or .txt format.")
            pep_all = pd.read_excel(pep_file)
        # pep_X: sparse data matrix
        pep_X = sparse.csr_matrix(pep_all.filter(regex='Abundance: F', axis=1).values).transpose()
        # pep_layers['mbr']: peptide MBR identification
        pep_layers_mbr = pep_all.filter(regex='Found in Sample', axis=1).values.transpose()
        # pep_var_names: peptide sequence with modifications
        pep_var_names = (pep_all['Annotated Sequence'] + np.where(pep_all['Modifications'].isna(), '', ' MOD:' + pep_all['Modifications'])).values
        # pep_obs_names: file names
        pep_obs_names = pep_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: (F\d+):')[0].values
        # pep_var: peptide metadata
        pep_var = pep_all.loc[:, 'Modifications':'Theo. MH+ [Da]']
        # prot_obs: sample typing from the column name
        pep_obs = pep_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: F\d+: (.+)$')[0].values
        pep_obs = pd.DataFrame(pep_obs, columns=['metadata'])['metadata'].str.split(',', expand=True).map(str.strip).astype('category')

        print(f"Number of files: {len(pep_obs_names)}")
        print(f"Number of peptides: {len(pep_var)}")
    else:
        pep_X = pep_layers_mbr = pep_var_names = pep_obs_names = pep_var = None

    if prot_file and pep_file:
        # -----------------------------
        # RS DATA
        # rs is in the form of a binary matrix, protein x peptide
        pep_prot_list = pep_all['Master Protein Accessions'].str.split('; ')
        mlb = MultiLabelBinarizer()
        rs = mlb.fit_transform(pep_prot_list)
        if prot_var_names is not None:
            index_dict = {protein: index for index, protein in enumerate(mlb.classes_)}
            reorder_indices = [index_dict[protein] for protein in prot_var_names]
            rs = rs[:, reorder_indices]
        print("RS matrix successfully computed")
    else:
        rs = None

    # ASSERTIONS
    # -----------------------------
    # check that all files overlap, and that the order is the same
    if prot_obs_names is not None and pep_obs_names is not None:
        assert set(pep_obs_names) == set(prot_obs_names), "The files in peptide and protein data must be the same"
    # -----------------------------
    # check if mlb.classes_ has overlap with prot_var
    if prot_file and pep_file:
        mlb_classes_set = set(mlb.classes_)
        prot_var_set = set(prot_var_names)

        if mlb_classes_set != prot_var_set:
            print("WARNING: Master proteins in the peptide matrix do not match proteins in the protein data, please check if files correspond to the same data.")
            print(f"Overlap: {len(mlb_classes_set & prot_var_set)}")
            print(f"Unique to peptide data: {mlb_classes_set - prot_var_set}")
            print(f"Unique to protein data: {prot_var_set - mlb_classes_set}")

    # pAnnData OBJECT - should be the same for all imports
    # -----------------------------
    pdata = pAnnData(prot_X, pep_X, rs)

    if prot_file:
        pdata.prot.obs = pd.DataFrame(prot_obs)
        pdata.prot.layers['X_mbr'] = prot_layers_mbr
        pdata.prot.layers['X_raw'] = prot_X
        pdata.prot.var = pd.DataFrame(prot_var)
        pdata.prot.obs_names = list(prot_obs_names)
        pdata.prot.var_names = list(prot_var_names)
        pdata.prot.obs.columns = obs_columns if obs_columns else list(range(len(prot_obs.columns)))
        pdata._append_history(f"Imported protein data from {prot_file}")

    if pep_file:
        pdata.pep.obs = pd.DataFrame(pep_obs)
        pdata.pep.layers['X_mbr'] = pep_layers_mbr
        pdata.pep.layers['X_raw'] = pep_X
        pdata.pep.var = pd.DataFrame(pep_var)
        pdata.pep.obs_names = list(pep_obs_names)
        pdata.pep.var_names = list(pep_var_names)
        pdata.pep.obs.columns = obs_columns if obs_columns else list(range(len(pep_obs.columns)))
        pdata._append_history(f"Imported peptide data from {pep_file}")

    pdata._update_summary()

    print("pAnnData object created. Use `print(pdata)` to view the object.")
    return pdata

def import_diann(report_file: Optional[str] = None, obs_columns: Optional[List[str]] = None, prot_value = 'PG.MaxLFQ', pep_value = 'Precursor.Translated'):
    if not report_file:
        raise ValueError("Importing from DIA-NN: report.tsv must be provided")
    print("--------------------------\nStarting import...\n--------------------------")

    print(f"Importing from {report_file}")
    report_all = pd.read_csv(report_file, sep='\t')
    report_all['Master.Protein'] = report_all['Protein.Group'].str.split(';')
    report_all = report_all.explode('Master.Protein')
    # -----------------------------
    # PROTEIN DATA
    # prot_X: sparse data matrix
    if prot_value is not 'PG.MaxLFQ':
        print("INFO: Protein value specified is not PG.MaxLFQ, please check if correct.")
    prot_X_pivot = report_all.pivot_table(index='Master.Protein', columns='Run', values=prot_value, aggfunc='first')
    prot_X = sparse.csr_matrix(prot_X_pivot.values).T
    # prot_var_names: protein names
    prot_var_names = prot_X_pivot.index.values
    # prot_obs: file names
    prot_obs_names = prot_X_pivot.columns.values

    # TO ADD: number of peptides detected?
    # prot_var: protein metadata
    prot_var_columns = ['Genes', 'Master.Protein']
    if 'First.Protein.Description' in report_all.columns:
        prot_var_columns.insert(0, 'First.Protein.Description')

    prot_var = report_all.loc[:, prot_var_columns].drop_duplicates(subset='Master.Protein').drop(columns='Master.Protein')
    # prot_obs: sample typing from the column name
    prot_obs = pd.DataFrame(prot_X_pivot.columns.values, columns=['Run'])['Run'].str.split('_', expand=True).rename(columns=dict(enumerate(obs_columns)))
    
    print(f"Number of files: {len(prot_obs_names)}")
    print(f"Number of proteins: {len(prot_var)}")

    # -----------------------------
    # PEPTIDE DATA
    # pep_X: sparse data matrix
    pep_X_pivot = report_all.pivot_table(index='Precursor.Id', columns='Run', values=pep_value, aggfunc='first')
    pep_X = sparse.csr_matrix(pep_X_pivot.values).T
    # pep_var_names: peptide sequence
    pep_var_names = pep_X_pivot.index.values
    # pep_obs_names: file names
    pep_obs_names = pep_X_pivot.columns.values
    # pep_var: peptide sequence with modifications
    pep_var_columns = ['Modified.Sequence', 'Stripped.Sequence', 'Precursor.Id']
    pep_var = report_all.loc[:, pep_var_columns].drop_duplicates(subset='Precursor.Id').drop(columns='Precursor.Id')
    # pep_obs: sample typing from the column name, same as prot_obs
    pep_obs = prot_obs

    print(f"Number of files: {len(pep_obs_names)}")
    print(f"Number of peptides: {len(pep_var)}")

    # -----------------------------
    # RS DATA
    # rs: protein x peptide relational data
    pep_prot_list = report_all.drop_duplicates(subset=['Precursor.Id'])['Protein.Group'].str.split(';')
    mlb = MultiLabelBinarizer()
    rs = mlb.fit_transform(pep_prot_list)
    index_dict = {protein: index for index, protein in enumerate(mlb.classes_)}
    reorder_indices = [index_dict[protein] for protein in prot_var_names]
    rs = rs[:, reorder_indices]
    print("RS matrix successfully computed")

    # -----------------------------
    # ASSERTIONS
    # -----------------------------
    # check that all files overlap, and that the order is the same
    if prot_obs_names is not None and pep_obs_names is not None:
        assert set(pep_obs_names) == set(prot_obs_names), "The files in peptide and protein data must be the same"
    # -----------------------------
    # check if mlb.classes_ has overlap with prot_var
    mlb_classes_set = set(mlb.classes_)
    prot_var_set = set(prot_var_names)

    if mlb_classes_set != prot_var_set:
        print("WARNING: Master proteins in the peptide matrix do not match proteins in the protein data, please check if files correspond to the same data.")
        print(f"Overlap: {len(mlb_classes_set & prot_var_set)}")
        print(f"Unique to peptide data: {mlb_classes_set - prot_var_set}")
        print(f"Unique to protein data: {prot_var_set - mlb_classes_set}")

    # pAnnData OBJECT - should be the same for all imports
    # -----------------------------

    pdata = pAnnData(prot_X, pep_X, rs)

    pdata.prot.obs = pd.DataFrame(prot_obs)
    pdata.prot.var = pd.DataFrame(prot_var)
    pdata.prot.obs_names = list(prot_obs_names)
    pdata.prot.var_names = list(prot_var_names)
    pdata.prot.obs.columns = obs_columns if obs_columns else list(range(len(prot_obs.columns)))

    pdata.pep.obs = pd.DataFrame(pep_obs)
    pdata.pep.var = pd.DataFrame(pep_var)
    pdata.pep.obs_names = list(pep_obs_names)
    pdata.pep.var_names = list(pep_var_names)
    pdata.pep.obs.columns = obs_columns if obs_columns else list(range(len(pep_obs.columns)))

    pdata._update_summary()
    pdata._append_history(f"Imported protein data from {report_file}, using {prot_value} as protein value and {pep_value} as peptide value.")
    print("pAnnData object created. Use `print(pdata)` to view the object.")

    return pdata
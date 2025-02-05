from encodings import normalize_encoding
from re import A
import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc

import copy
import warnings

from scipy import sparse
from scipy.stats import variation, ttest_ind, mannwhitneyu, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer, normalize, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer

import umap.umap_ as umap

import seaborn as sns
import matplotlib.pyplot as plt

from pandas.testing import assert_frame_equal

from scviz import utils
from scviz import setup

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
    stats : Dict
        Dictionary of differential expression results.
    
    !TODO:
    - Decide whether classes or class_types
        
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
        self._stats = {}

    # -----------------------------
    # SETTERS/GETTERS    
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
    
    @property
    def stats(self):
        return self._stats

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

    @stats.setter
    def stats(self, value):
        self._stats = value

    # -----------------------------
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

    def _update_summary(self):
        if self.prot is not None:
            # note: missing values is 1-protein_quant
            self.prot.obs['protein_quant'] = np.sum(~np.isnan(self.prot.X.toarray()), axis=1) / self.prot.X.shape[1]
            self.prot.obs['protein_count'] = np.sum(~np.isnan(self.prot.X.toarray()), axis=1)
            self.prot.obs['protein_abundance_sum'] = np.nansum(self.prot.X.toarray(), axis=1)
            
            if 'X_mbr' in self.prot.layers:
                self.prot.obs['mbr_count'] = (self.prot.layers['X_mbr'] == 'Peak Found').sum(axis=1)
                self.prot.obs['high_count'] = (self.prot.layers['X_mbr'] == 'High').sum(axis=1)

        if self.pep is not None:
            # note: missing values is 1-peptide_quant
            self.pep.obs['peptide_quant'] = np.sum(~np.isnan(self.pep.X.toarray()), axis=1) / self.pep.X.shape[1]
            self.pep.obs['peptide_count'] = np.sum(~np.isnan(self.pep.X.toarray()), axis=1)
            self.pep.obs['peptide_abundance_sum'] = np.nansum(self.pep.X.toarray(), axis=1)

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

    def print_history(self):
        formatted_history = "\n".join(f"{i}: {action}" for i, action in enumerate(self._history, 1))
        print("-------------------------------\nHistory:\n-------------------------------\n"+formatted_history)

    # -----------------------------
    # TESTS FUNCTIONS

    def _check_data(self, on):
        # check if protein or peptide data exists
        if on not in ['protein', 'peptide']:
            raise ValueError("Invalid input: on must be either 'protein' or 'peptide'.")
        elif on == 'protein' and self.prot is None:
            raise ValueError("No protein data found in AnnData object.")
        elif on == 'peptide' and self.pep is None:
            raise ValueError("No peptide data found in AnnData object.")
        else:
            return True

    def _check_rankcol(self, on = 'protein', class_values = None):
        # check if average and rank columns exist for the specified class values
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


    # this is more of a hardcorded filter function (needs a hardcoded query), util filter function is more general soft-coded filter function
    # maybe call this filter_query, and leave the other as filter
    # TODO: in docs, show how both functions can be used for the same filtering task
    def filter(self, condition = None, return_copy = True, file_list = None):
        # TODO: add docstring, add example of file_list usage
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
    def cv(self, classes = None, on = 'protein', layer = "X", debug = False):
        if not self._check_data(on):
            pass

        adata = self.prot if on == 'protein' else self.pep
        classes_list = utils.get_classlist(adata, classes)

        for j, class_value in enumerate(classes_list):
            if classes is None:
                values = class_value.split('_')
                print(f'Classes: {classes}, Values: {values}') if debug else None
                data_filtered = utils.filter(adata, classes, values, suppress_warnings=True)
            elif isinstance(classes, str):
                print(f'Class: {classes}, Value: {class_value}') if debug else None
                data_filtered = utils.filter(adata, classes, class_value, suppress_warnings=True)
            elif isinstance(classes, list):
                values = class_value.split('_')
                print(f'Classes: {classes}, Values: {values}') if debug else None
                data_filtered = utils.filter(adata, classes, values, suppress_warnings=True)

            cv_data = data_filtered.X.toarray() if layer == "X" else data_filtered.layers[layer].toarray() if layer in data_filtered.layers else None
            if cv_data is None:
                raise ValueError(f"Layer '{layer}' not found in adata.layers.")

            adata.var['CV: '+ class_value] = variation(cv_data, axis=0)

        self._history.append(f"{on}: Coefficient of Variation (CV) calculated for {layer} data by {classes}. E.g. CV stored in var['CV: {class_value}'].")

    # FC method: specify mean, prot pairwise median, or pep pairwise median
    # TODO: implement layer support
    def de(self, class_type, values, method = 'ttest', layer = "X", pval=0.05, log2fc=1):
        """
        Calculate differential expression (DE) of proteins across different groups.

        This function calculates the DE of proteins across different groups. The cases to compare can be specified, and the method to use for DE can be specified as well.

        Args:
            self (pAnnData): The pAnnData object containing the protein data.
            class_type (str): The class type to use for selecting samples. E.g. 'cell_type'.
            values (list of list of str): The values to select for within the class_type. E.g. [['wt', 'kd'], ['control', 'treatment']].
            method (str, optional): The method to use for DE. Default is 'ttest'. Other methods include 'mannwhitneyu', 'wilcoxon', 'chi2', and 'fisher'. !TODO: implement pairwise prot median, pairwise pep median.

        Returns:
            df_stats (pandas.DataFrame): A DataFrame containing the DE statistics for each protein.

        Raises:
            ValueError: If the number of cases is not exactly two.

        Example:
            >>> from scviz import utils as scutils
            >>> stats_sc_20000 = pdata.de(['cell_type','size'], [['cortex', 'sc'], ['cortex', '20000']])
        """

        # this is for case 1/case 2 comparison!
        # make sure only two cases are given
        if len(values) != 2:
            raise ValueError('Please provide exactly two cases to compare.')

        pdata_case1 = utils.filter(self, class_type, values[0], exact_cases=True)
        pdata_case2 = utils.filter(self, class_type, values[1], exact_cases=True)

        # TODO: Need to implement pairwise differential expression...
        # if on == 'protein':
        #     abundance_case1 = pdata_case1.prot
        #     abundance_case2 = pdata_case2.prot
        # elif on == 'peptide':
        #     abundance_case1 = pdata_case1.pep
        #     abundance_case2 = pdata_case2.pep

        abundance_case1 = pdata_case1.prot
        abundance_case2 = pdata_case2.prot

        n1 = abundance_case1.shape[0]
        n2 = abundance_case2.shape[0]

        group1_string = '_'.join(values[0]) if isinstance(values[0], list) else values[0]
        group2_string = '_'.join(values[1]) if isinstance(values[1], list) else values[1]

        comparison_string = f'{group1_string} vs {group2_string}'

        # create a dataframe for stats
        df_stats = pd.DataFrame(index=abundance_case1.var_names, columns=[group1_string,group2_string,'log2fc', 'p_value', 'test_statistic'])
        df_stats['Genes'] = abundance_case1.var['Genes']
        df_stats[group1_string] = np.mean(abundance_case1.X.toarray(), axis=0)
        df_stats[group2_string] = np.mean(abundance_case2.X.toarray(), axis=0)
        df_stats['log2fc'] = np.log2(np.divide(np.mean(abundance_case1.X.toarray(), axis=0), np.mean(abundance_case2.X.toarray(), axis=0)))

        if method == 'ttest':
            for protein in range(0, abundance_case1.shape[1]):
                t_test = ttest_ind(abundance_case1.X.toarray()[:,protein], abundance_case2.X.toarray()[:,protein])
                df_stats['p_value'].iloc[protein] = t_test.pvalue
                df_stats['test_statistic'].iloc[protein] = t_test.statistic
        elif method == 'mannwhitneyu':
            for row in range(0, len(abundance_case1)):
                mwu = mannwhitneyu(abundance_case1.iloc[row,0:n1-1].dropna().values, abundance_case2.iloc[row,0:n2-1].dropna().values)
                df_stats['p_value'].iloc[row] = mwu.pvalue
                df_stats['test_statistic'].iloc[row] = mwu.statistic
        elif method == 'wilcoxon':
            for row in range(0, len(abundance_case1)):
                w, p = wilcoxon(abundance_case1.iloc[row,0:n1-1].dropna().values, abundance_case2.iloc[row,0:n2-1].dropna().values)
                df_stats['p_value'].iloc[row] = p
                df_stats['test_statistic'].iloc[row] = w

        df_stats['p_value'] = df_stats['p_value'].replace([np.inf, -np.inf], np.nan)
        non_nan_mask = df_stats['p_value'].notna()

        df_stats.loc[non_nan_mask, '-log10(p_value)'] = -np.log10(df_stats.loc[non_nan_mask, 'p_value'])
        df_stats.loc[non_nan_mask, 'significance_score'] = df_stats.loc[non_nan_mask, '-log10(p_value)'] * df_stats.loc[non_nan_mask, 'log2fc']

        df_stats.loc[non_nan_mask, 'significance'] = 'not significant'
        df_stats.loc[non_nan_mask & (df_stats['p_value'] < pval) & (df_stats['log2fc'] > log2fc), 'significance'] = 'upregulated'
        df_stats.loc[non_nan_mask & (df_stats['p_value'] < pval) & (df_stats['log2fc'] < -log2fc), 'significance'] = 'downregulated'

        df_stats = df_stats.dropna(subset=['p_value', 'log2fc', 'significance'])
        df_stats['significance'] = pd.Categorical(df_stats['significance'], categories=['upregulated', 'downregulated', 'not significant'], ordered=True)
        df_stats = df_stats.sort_values(by='significance')

        self._stats[comparison_string] = df_stats
        print(f'Differential expression calculated for {class_type} {values} using {method}. DE statistics stored in .stats["{comparison_string}"].')
        self._append_history(f"prot: Differential expression calculated for {class_type} {values} using {method}. DE statistics stored in .stats['{comparison_string}'].")

        return df_stats

    # TODO: Need to figure out how to make this interface with plot functions, probably do reordering by each class_value within the loop?
    def rank(self, classes = None, on = 'protein', layer = "X"):
        if not self._check_data(on):
            pass

        adata = self.prot if on == 'protein' else self.pep
        classes_list = utils.get_classlist(adata, classes)
        
        for j, class_value in enumerate(classes_list):
            if classes is None:
                values = class_value.split('_')
                print(f'Classes: {classes}, Values: {values}')
                rank_data = utils.filter(adata, classes, values, suppress_warnings=True)
            elif isinstance(classes, str):
                print(f'Class: {classes}, Value: {class_value}')
                rank_data = utils.filter(adata, classes, class_value, suppress_warnings=True)
            elif isinstance(classes, list):
                values = class_value.split('_')
                print(f'Classes: {classes}, Values: {values}')
                rank_data = utils.filter(adata, classes, values, suppress_warnings=True)

            rank_df = rank_data.to_df().transpose()
            rank_df['Average: '+class_value] = np.nanmean(rank_data.X.toarray(), axis=0)
            rank_df['Stdev: '+class_value] = np.nanstd(rank_data.X.toarray(), axis=0)
            rank_df.sort_values(by=['Average: '+class_value], ascending=False, inplace=True)
            rank_df['Rank: '+class_value] = np.where(rank_df['Average: '+class_value].isna(), np.nan, np.arange(1, len(rank_df) + 1))

            # revert back to original data order (need to do so, because we sorted rank_df)
            sorted_indices = rank_df.index
            rank_df = rank_df.loc[adata.var.index]
            adata.var['Average: ' + class_value] = rank_df['Average: ' + class_value]
            adata.var['Stdev: ' + class_value] = rank_df['Stdev: ' + class_value]
            adata.var['Rank: ' + class_value] = rank_df['Rank: ' + class_value]
            rank_df = rank_df.reindex(sorted_indices)
            

        self._history.append(f"{on}: Ranked {layer} data. Ranking, average and stdev stored in var.")

    def impute(self, classes = None, layer = "X", method = 'min', on = 'protein', set_X = True, **kwargs):
        '''Function for imputation, imputes data across samples and stores it in the pdata layer X_impute_method.
        Unfortunately, the imputers in scikit-learn only impute columns, not rows, which means ColumnTransformer+SimpleImputers won't work.
        KNN to be implemented later.

        Args:
            classes (list): List of classes to impute.
            layer (str): Layer to impute.
            method (str): Imputation method.
            on (str): 'protein' or 'peptide'.
        '''
        if not self._check_data(on):
            pass

        adata = self.prot if on == 'protein' else self.pep
        if layer != "X" and layer not in adata.layers:
            raise ValueError(f"Layer {layer} not found in .{on}.")

        impute_funcs = {
            'mean': np.nanmean,
            'median': np.nanmedian,
            'min': np.nanmin
        }

        if method not in impute_funcs:
            raise ValueError(f"Unknown method: {method}")
        
        impute_func = impute_funcs.get(method)
        impute_data = adata.layers[layer] if layer != "X" else adata.X
        layer_name = 'X_impute_' + method
        was_sparse = sparse.issparse(impute_data)

        if was_sparse:
            impute_data = impute_data.toarray()

        if classes is None:
            impute_data[np.isnan(impute_data)] = np.take(impute_func(impute_data, axis=0), np.where(np.isnan(impute_data))[1])

            print(f"Imputed data across all samples using {method}. New data stored in `{layer_name}`.")
            self._history.append(f"{on}: Imputed layer {layer} using {method}. Imputed data stored in `{layer_name}`.")

        else:
            sample_names = utils.get_samplenames(adata, classes)
            unique_identifiers = list(set(sample_names))
            indices_dict = {identifier: [i for i, sample in enumerate(sample_names) if sample == identifier] for identifier in unique_identifiers}

            for identifier, indices in indices_dict.items():
                subset_data = impute_data[indices,:]
                subset_data[np.isnan(subset_data)] = np.take(impute_func(subset_data, axis=0), np.where(np.isnan(subset_data))[1])
                impute_data[indices,:] = subset_data

            print(f"{on}: Imputed based on class(es): {classes} - {unique_identifiers} using {method}. New data stored in `{layer_name}`.")
            self._history.append(f"{on}: Imputed layer {layer} based on class(es): {classes} - {unique_identifiers} using {method}. Imputed data stored in `{layer_name}`.")

        if was_sparse:
            adata.layers[layer_name] = sparse.csr_matrix(impute_data)
        else:
            adata.layers[layer_name] = impute_data

        if set_X:
            self.set_X(layer = layer_name, on = on)

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

        if 'pca' not in adata.uns:
            print("PCA not found in AnnData object. Running PCA with default settings.")
            self.pca(on = on, layer = layer)

        sc.pp.neighbors(adata, **kwargs)

        self._append_history(f'{on}: Neighbors fitted on {layer}, stored in obs["distances"] and obs["connectivities"]')
        print(f'{on}: Neighbors fitted on {layer} and and stored in obs["distances"] and obs["connectivities"]')

    def leiden(self, on = 'protein', layer = "X", **kwargs):
        # uses sc.tl.leiden with default resolution of 0.25
        if not self._check_data(on):
            pass

        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        if 'neighbors' not in adata.uns:
            print("Neighbors not found in AnnData object. Running neighbors with default settings.")
            self.neighbor(on = on, layer = layer)

        if layer == "X":
            # do nothing
            pass
        elif layer in adata.layers.keys():
            self.set_X(layer = layer, on = on)

        sc.tl.leiden(adata, **kwargs)

        self._append_history(f'{on}: Leiden clustering fitted on {layer}, stored in obs["leiden"]')
        print(f'{on}: Leiden clustering fitted on {layer} and and stored in obs["leiden"]')

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

    def nanmissingvalues(self, on = 'protein', limit = 0.5):
        # sets columns (proteins and peptides) with > 0.5 missing values to NaN across all samples
        if not self._check_data(on):
            pass

        if on == 'protein':
            adata = self.prot

        elif on == 'peptide':
            adata = self.pep

        missing_proportion = np.isnan(adata.X.toarray()).mean(axis=0)
        columns_to_nan = missing_proportion > limit
        adata.X[:, columns_to_nan] = np.nan

        if on == 'protein':
            self.prot = adata
        elif on == 'peptide':
            self.pep = adata

    def normalize(self, classes = None, layer = "X", method = 'sum', on = 'protein', set_X = True, **kwargs):  
        if not self._check_data(on):
            pass

        adata = self.prot if on == 'protein' else self.pep
        if layer != "X" and layer not in adata.layers:
            raise ValueError(f"Layer {layer} not found in .{on}.")

        normalize_funcs = ['sum', 'median', 'mean', 'max', 'reference_feature', 'robust_scale', 'quantile_transform']

        if method not in normalize_funcs:
            raise ValueError(f"Unknown method: {method}")
        
        normalize_data = adata.layers[layer] if layer != "X" else adata.X
        layer_name = 'X_norm_' + method
        was_sparse = sparse.issparse(normalize_data)

        if was_sparse:
            normalize_data = normalize_data.toarray()

        if classes is None:
            normalize_data = _normalize_helper(normalize_data, method, **kwargs)
            
            print(f"Normalized data across all samples using {method}. New data stored in `{layer_name}`.")
            self._history.append(f"{on}: Normalized layer {layer} using {method}. Imputed data stored in `{layer_name}`.")
                  
        if was_sparse:
            adata.layers[layer_name] = sparse.csr_matrix(normalize_data)
        else:
            adata.layers[layer_name] = normalize_data

        if set_X:
            self.set_X(layer = layer_name, on = on)

def _normalize_helper(data, method, **kwargs):
    if method == 'sum':
        # norm by sum: scale each row s.t. sum of each row is the same as the max sum
        data_norm = data * (np.nansum(data, axis=1).max() / (np.nansum(data, axis=1)))[:, None]

    elif method == 'median':
        # norm by median: scale each row s.t. median of each row is the same as the max median
        data_norm = data * (np.nanmedian(data, axis=1).max() / (np.nanmedian(data, axis=1)))[:, None]

    elif method == 'mean':
        # norm by mean: scale each row s.t. mean of each row is the same as the max mean
        data_norm = data * (np.nanmean(data, axis=1).max() / (np.nanmean(data, axis=1)))[:, None]

    elif method == 'max':
        # norm by max: scale each row s.t. max value of each row is the same as the max value
        data_norm = data * (np.nanmax(data, axis=1).max() / (np.nanmax(data, axis=1)))[:, None]

    elif method == 'reference_feature':
        # norm by reference feature: scale each row s.t. the reference column is the same across all rows (scale to max value of reference column)
        reference_columns = kwargs.get('reference_columns', [2])
        scaling_factors = np.nanmean(np.nanmax(data[:, reference_columns], axis=0) / (data[:, reference_columns]), axis=1)

        nan_rows = np.where(np.isnan(scaling_factors))[0]
        if nan_rows.size > 0:
            print(f"Rows ({', '.join(map(str, nan_rows))}) were normalized by mean scaling factor because reference feature was missing. Suggest imputation on reference feature by class and re-normalize.")

        scaling_factors = np.where(np.isnan(scaling_factors), np.nanmean(scaling_factors), scaling_factors)
        data_norm = data * scaling_factors[:, None]

    elif method == 'robust_scale':
        # norm by robust_scale: Center to the median and component wise scale according to the interquartile range. See sklearn.preprocessing.robust_scale for more information.
        from sklearn.preprocessing import robust_scale
        data_norm = robust_scale(data, axis=1)

    elif method == 'quantile_transform':
        # norm by quantile_transform: Transform features using quantiles information. See sklearn.preprocessing.quantile_transform for more information.
        from sklearn.preprocessing import quantile_transform
        data_norm = quantile_transform(data, axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    return data_norm

def import_proteomeDiscoverer(prot_file: Optional[str] = None, pep_file: Optional[str] = None, obs_columns: Optional[List[str]] = ['sample']):
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
        prot_var.rename(columns={'Gene Symbol': 'Genes'}, inplace=True)
        # prot_obs_names: file names
        prot_obs_names = prot_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: (F\d+):')[0].values
        # prot_obs: sample typing from the column name, drop column if all 'n/a'
        prot_obs = prot_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: F\d+: (.+)$')[0].values
        prot_obs = pd.DataFrame(prot_obs, columns=['metadata'])['metadata'].str.split(',', expand=True).map(str.strip).astype('category')
        if (prot_obs == "n/a").all().any():
            print("WARNING: Found columns with all 'n/a'. Dropping these columns.")
            prot_obs = prot_obs.loc[:, ~(prot_obs == "n/a").all()]

        print(f"Number of files: {len(prot_obs_names)}")
        print(f"Number of proteins: {len(prot_var)}")
        print(f"Number of obs: {len(prot_obs.columns)}")
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
        # prot_obs: sample typing from the column name, drop column if all 'n/a'
        pep_obs = pep_all.filter(regex='Abundance: F', axis=1).columns.str.extract('Abundance: F\d+: (.+)$')[0].values
        pep_obs = pd.DataFrame(pep_obs, columns=['metadata'])['metadata'].str.split(',', expand=True).map(str.strip).astype('category')
        if (pep_obs == "n/a").all().any():
            print("WARNING: Found columns with all 'n/a'. Dropping these columns.")
            pep_obs = pep_obs.loc[:, ~(pep_obs == "n/a").all()]


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

def import_diann(report_file: Optional[str] = None, obs_columns: Optional[List[str]] = None, prot_value = 'PG.MaxLFQ', pep_value = 'Precursor.Normalised', prot_var_columns = ['Genes', 'Master.Protein'], pep_var_columns = ['Genes', 'Protein.Group', 'Precursor.Charge','Modified.Sequence', 'Stripped.Sequence', 'Precursor.Id', 'All Mapped Proteins', 'All Mapped Genes']):
    if not report_file:
        raise ValueError("Importing from DIA-NN: report.tsv or report.parquet must be provided")
    print("--------------------------\nStarting import...\n--------------------------")

    print(f"Importing from {report_file}")
    # if csv, then use pd.read_csv, if parquet then use pd.read_parquet('example_pa.parquet', engine='pyarrow')
    if report_file.endswith('.tsv'):
        report_all = pd.read_csv(report_file, sep='\t')
    elif report_file.endswith('.parquet'):
        report_all = pd.read_parquet(report_file, engine='pyarrow')
    report_all['Master.Protein'] = report_all['Protein.Group'].str.split(';')
    report_all = report_all.explode('Master.Protein')
    # -----------------------------
    # PROTEIN DATA
    # prot_X: sparse data matrix
    if prot_value is not 'PG.MaxLFQ':
        if report_file.endswith('.tsv') and prot_value == 'PG.Quantity':
            # throw an error that DIA-NN version >2.0 does not have PG.quantity
            raise ValueError("Reports generated with DIA-NN version >2.0 do not contain PG.Quantity values, please use PG.MaxLFQ .")
        else:
            print("INFO: Protein value specified is not PG.MaxLFQ, please check if correct.")
    prot_X_pivot = report_all.pivot_table(index='Master.Protein', columns='Run', values=prot_value, aggfunc='first', sort=False)
    prot_X = sparse.csr_matrix(prot_X_pivot.values).T
    # prot_var_names: protein names
    prot_var_names = prot_X_pivot.index.values
    # prot_obs: file names
    prot_obs_names = prot_X_pivot.columns.values

    # prot_var: protein metadata (default: Genes, Master.Protein)
    if 'First.Protein.Description' in report_all.columns:
        prot_var_columns.insert(0, 'First.Protein.Description')

    existing_prot_var_columns = [col for col in prot_var_columns if col in report_all.columns]
    missing_columns = set(prot_var_columns) - set(existing_prot_var_columns)

    if missing_columns:
        warnings.warn(
            f"Warning: The following columns are missing: {', '.join(missing_columns)}. "
        )

    prot_var = report_all.loc[:, existing_prot_var_columns].drop_duplicates(subset='Master.Protein').drop(columns='Master.Protein')
    # prot_obs: sample typing from the column name
    if obs_columns is None:
        num_files = len(prot_X_pivot.columns)
        prot_obs = pd.DataFrame({'File': range(1, num_files + 1)})
    else:
        prot_obs = pd.DataFrame(prot_X_pivot.columns.values, columns=['Run'])['Run'].str.split('_', expand=True).rename(columns=dict(enumerate(obs_columns)))
    
    print(f"Number of files: {len(prot_obs_names)}")
    print(f"Number of proteins: {len(prot_var)}")

    # -----------------------------
    # PEPTIDE DATA
    # pep_X: sparse data matrix
    pep_X_pivot = report_all.pivot_table(index='Precursor.Id', columns='Run', values=pep_value, aggfunc='first', sort=False)
    pep_X = sparse.csr_matrix(pep_X_pivot.values).T
    # pep_var_names: peptide sequence
    pep_var_names = pep_X_pivot.index.values
    # pep_obs_names: file names
    pep_obs_names = pep_X_pivot.columns.values
    # pep_var: peptide sequence with modifications (default: Genes, Protein.Group, Precursor.Charge, Modified.Sequence, Stripped.Sequence, Precursor.Id, All Mapped Proteins, All Mapped Genes)
    existing_pep_var_columns = [col for col in pep_var_columns if col in report_all.columns]
    missing_columns = set(pep_var_columns) - set(existing_pep_var_columns)

    if missing_columns:
        warnings.warn(
            f"Warning: The following columns are missing: {', '.join(missing_columns)}. "
            "Consider running analysis in the newer version of DIA-NN (1.8.1). "
            "Peptide-protein mapping may differ."
        )
    
    pep_var = report_all.loc[:, existing_pep_var_columns].drop_duplicates(subset='Precursor.Id').drop(columns='Precursor.Id')
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

    # TO ADD: number of peptides per protein
    # number of proteins per peptide
    # make columns isUnique for peptide
    sum_values = np.sum(rs, axis=1)

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
    pdata.prot.layers['X_raw'] = prot_X

    pdata.pep.obs = pd.DataFrame(pep_obs)
    pdata.pep.var = pd.DataFrame(pep_var)
    pdata.pep.obs_names = list(pep_obs_names)
    pdata.pep.var_names = list(pep_var_names)
    pdata.pep.obs.columns = obs_columns if obs_columns else list(range(len(pep_obs.columns)))
    pdata.pep.layers['X_raw'] = pep_X

    pdata._update_summary()
    pdata._append_history(f"Imported protein data from {report_file}, using {prot_value} as protein value and {pep_value} as peptide value.")
    print("pAnnData object created. Use `print(pdata)` to view the object.")

    return pdata
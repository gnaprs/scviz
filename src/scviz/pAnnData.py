from encodings import normalize_encoding
import re
import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import datetime

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
    
    @property
    def metadata(self):
        if self.prot is not None and 'metadata' in self.prot.uns:
            return self.prot.uns['metadata']
        elif self.pep is not None and 'metadata' in self.pep.uns:
            return self.pep.uns['metadata']
        return {}

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
    def _set_RS(self, rs, debug=False, validate=True):
        """
        Internal method to set the RS (protein × peptide) mapping matrix.
        Transposes the input if it's in peptide × protein format.

        Parameters:
        - rs (array or sparse matrix): The new RS matrix
        - debug (bool): Print debug info
        - validate (bool): If True (default), check that RS shape matches .prot and .pep
        """
        if debug:
            print(f"Setting rs matrix with dimensions {rs.shape}")

        # Only validate if requested (e.g. for external .rs = ... use)
        if validate:
            prot_n = self.prot.shape[1] if self.prot is not None else None
            pep_n = self.pep.shape[1] if self.pep is not None else None
            rs_shape = rs.shape

            valid_prot_pep = (prot_n is None or rs_shape[0] == prot_n) and (pep_n is None or rs_shape[1] == pep_n)
            valid_pep_prot = (prot_n is None or rs_shape[1] == prot_n) and (pep_n is None or rs_shape[0] == pep_n)

            if not (valid_prot_pep or valid_pep_prot):
                raise ValueError(
                    f"❌ RS shape {rs_shape} does not match expected protein × peptide "
                    f"({prot_n} × {pep_n}) or peptide × protein ({pep_n} × {prot_n})."
                )

            # Transpose if necessary
            if self.prot is not None and rs_shape[0] != prot_n:
                if debug:
                    print("↔️  Transposing RS matrix to match protein × peptide format")
                rs = rs.T

        # Always store as sparse
        self._rs = sparse.csr_matrix(rs)

        if debug:
            nnz = self._rs.nnz
            total = self._rs.shape[0] * self._rs.shape[1]
            sparsity = 100 * (1 - nnz / total)
            print(f"✅ RS matrix set: {self._rs.shape} (proteins × peptides), sparsity: {sparsity:.2f}%")



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

    # def _update_summary(self):
    #     if self.prot is not None:
    #         # note: missing values is 1-protein_quant
    #         self.prot.obs['protein_quant'] = np.sum(~np.isnan(self.prot.X.toarray()), axis=1) / self.prot.X.shape[1]
    #         self.prot.obs['protein_count'] = np.sum(~np.isnan(self.prot.X.toarray()), axis=1)
    #         self.prot.obs['protein_abundance_sum'] = np.nansum(self.prot.X.toarray(), axis=1)
            
    #         if 'X_mbr' in self.prot.layers:
    #             self.prot.obs['mbr_count'] = (self.prot.layers['X_mbr'] == 'Peak Found').sum(axis=1)
    #             self.prot.obs['high_count'] = (self.prot.layers['X_mbr'] == 'High').sum(axis=1)

    #     if self.pep is not None:
    #         # note: missing values is 1-peptide_quant
    #         self.pep.obs['peptide_quant'] = np.sum(~np.isnan(self.pep.X.toarray()), axis=1) / self.pep.X.shape[1]
    #         self.pep.obs['peptide_count'] = np.sum(~np.isnan(self.pep.X.toarray()), axis=1)
    #         self.pep.obs['peptide_abundance_sum'] = np.nansum(self.pep.X.toarray(), axis=1)

    #         if 'X_mbr' in self.pep.layers:
    #             self.pep.obs['mbr_count'] = (self.pep.layers['X_mbr'] == 'Peak Found').sum(axis=1)
    #             self.pep.obs['high_count'] = (self.pep.layers['X_mbr'] == 'High').sum(axis=1)
        
    #     if self.prot is not None:
    #         self._summary = self.prot.obs.copy()
    #         if self.pep is not None:
    #             for col in self.pep.obs.columns:
    #                 if col not in self._summary.columns:
    #                     self._summary[col] = self.pep.obs[col]
    #     else:
    #         self._summary = self.pep.obs.copy()

    #     self._previous_summary = self._summary.copy()
        
    # def _update_obs(self):
    #     # function to update obs with summary data (if user edited summary data)
    #     if not self._has_data():
    #         return

    #     def update_obs_with_summary(obs, summary, ignore_keyword):
    #         ignored_columns = []
    #         for col in summary.columns:
    #             if ignore_keyword in col:
    #                 ignored_columns.append(col)
    #                 continue
    #             obs[col] = summary[col]
    #         return ignored_columns
            

    #     if self.prot is not None:
    #         if not self.prot.obs.index.equals(self._summary.index):
    #             raise ValueError("Index of summary does not match index of prot.obs")
    #         ignored_columns_prot = update_obs_with_summary(self.prot.obs, self._summary, "pep")
    #     else:
    #         ignored_columns_prot = None
    #     if self.pep is not None:
    #         if not self.pep.obs.index.equals(self._summary.index):
    #             raise ValueError("Index of summary does not match index of pep.obs")
    #         ignored_columns_pep = update_obs_with_summary(self.pep.obs, self._summary, "prot")
    #     else:
    #         ignored_columns_pep = None

    #     history_statement = "Updated obs with summary data. "
    #     if ignored_columns_prot:
    #         history_statement += f"Ignored columns in prot.obs: {', '.join(ignored_columns_prot)}. "
    #     if ignored_columns_pep:
    #         history_statement += f"Ignored columns in pep.obs: {', '.join(ignored_columns_pep)}. "
    #     self._history.append(history_statement)

    def _update_metrics(self):
        """Compute per-sample and RS-derived metrics for prot and pep data."""
        
        if self.prot is not None:
            X = self.prot.X.toarray()
            self.prot.obs['protein_quant'] = np.sum(~np.isnan(X), axis=1) / X.shape[1]
            self.prot.obs['protein_count'] = np.sum(~np.isnan(X), axis=1)
            self.prot.obs['protein_abundance_sum'] = np.nansum(X, axis=1)

            if 'X_mbr' in self.prot.layers:
                self.prot.obs['mbr_count'] = (self.prot.layers['X_mbr'] == 'Peak Found').sum(axis=1)
                self.prot.obs['high_count'] = (self.prot.layers['X_mbr'] == 'High').sum(axis=1)

        if self.pep is not None:
            X = self.pep.X.toarray()
            self.pep.obs['peptide_quant'] = np.sum(~np.isnan(X), axis=1) / X.shape[1]
            self.pep.obs['peptide_count'] = np.sum(~np.isnan(X), axis=1)
            self.pep.obs['peptide_abundance_sum'] = np.nansum(X, axis=1)

            if 'X_mbr' in self.pep.layers:
                self.pep.obs['mbr_count'] = (self.pep.layers['X_mbr'] == 'Peak Found').sum(axis=1)
                self.pep.obs['high_count'] = (self.pep.layers['X_mbr'] == 'High').sum(axis=1)

        # RS metrics for prot.var
        if self.rs is not None and self.prot is not None:
            rs = self.rs.toarray() if sparse.issparse(self.rs) else self.rs
            peptides_per_protein = rs.sum(axis=1).astype(int)
            unique_mask = rs.sum(axis=0) == 1
            unique_counts = rs[:, unique_mask].sum(axis=1).astype(int)
            self.prot.var['peptides_per_protein'] = peptides_per_protein
            self.prot.var['unique_peptides'] = unique_counts

    def _update_summary_metrics(self, unique_peptide_thresh=2):
        """
        Add RS-derived per-sample metrics to ._summary.
        Currently adds 'unique_protein_count' (e.g., ≥2 unique peptides per protein).
        """
        if (
            self.rs is not None and
            self.prot is not None and
            hasattr(self, '_summary') and
            'unique_peptides' in self.prot.var.columns
        ):
            unique_mask = self.prot.var['unique_peptides'] >= unique_peptide_thresh
            quant_matrix = self.prot.X.toarray()
            high_conf_matrix = quant_matrix[:, unique_mask]
            high_conf_count = np.sum(~np.isnan(high_conf_matrix), axis=1)
            self._summary['unique_pep2_protein_count'] = high_conf_count

    def _merge_obs(self):
        """
        Merge .prot.obs and .pep.obs into a single summary DataFrame.
        Shared columns (e.g. 'gradient', 'condition') are kept from .prot by default.
        """
        if self.prot is not None:
            summary = self.prot.obs.copy()
            if self.pep is not None:
                for col in self.pep.obs.columns:
                    if col not in summary.columns:
                        summary[col] = self.pep.obs[col]
        elif self.pep is not None:
            summary = self.pep.obs.copy()
        else:
            summary = pd.DataFrame()

        self._summary = summary
        self._previous_summary = summary.copy()

    def _push_summary_to_obs(self, skip_if_contains='pep', verbose=True):
        """
        Push changes from .summary back into .prot.obs and .pep.obs.
        Columns containing `skip_if_contains` are ignored for .prot; same for 'prot' in .pep.
        """
        if not self._has_data():
            return

        def update_obs_with_summary(obs, summary, skip_if_contains):
            skipped = []
            for col in summary.columns:
                if skip_if_contains in col:
                    skipped.append(col)
                    continue
                obs[col] = summary[col]
            return skipped

        if self.prot is not None:
            if not self.prot.obs.index.equals(self._summary.index):
                raise ValueError("Mismatch: .summary and .prot.obs have different sample indices.")
            skipped_prot = update_obs_with_summary(self.prot.obs, self._summary, skip_if_contains)
        else:
            skipped_prot = None

        if self.pep is not None:
            if not self.pep.obs.index.equals(self._summary.index):
                raise ValueError("Mismatch: .summary and .pep.obs have different sample indices.")
            skipped_pep = update_obs_with_summary(self.pep.obs, self._summary, skip_if_contains='prot')
        else:
            skipped_pep = None

        msg = "Pushed summary values back to obs. "
        if skipped_prot:
            msg += f"Skipped for prot: {', '.join(skipped_prot)}. "
        if skipped_pep:
            msg += f"Skipped for pep: {', '.join(skipped_pep)}. "

        self._append_history(msg)
        if verbose:
            print(msg)

    def update_summary(self, recompute=True, sync_back=False, verbose=True):
        """
        Update the .summary dataframe from .obs (and optionally push changes back down).

        Parameters:
        - recompute (bool): If True, re-calculate protein/peptide stats.
        - sync_back (bool): If True, push edited .summary values back to .prot.obs / .pep.obs. False by default, as .summary is derived.
        - verbose (bool): If True, print action messages.
        """
        if recompute:
            self._update_metrics()
        self._merge_obs()
        self._update_summary_metrics()
        if sync_back:
            self._push_summary_to_obs(verbose=verbose)
        elif verbose:
            mode = []
            if recompute: mode.append("recompute")
            if sync_back: mode.append("sync_back")
            print(f"[update_summary] → Mode: {', '.join(mode) or 'noop'}")

    def _update_summary(self):
        print("⚠️  Legacy _update_summary() called — consider switching to update_summary()")
        self.update_summary(recompute=True, sync_back=False, verbose=False)

    def _append_history(self, action):
        self._history.append(action)

    def print_history(self):
        formatted_history = "\n".join(f"{i}: {action}" for i, action in enumerate(self._history, 1))
        print("-------------------------------\nHistory:\n-------------------------------\n"+formatted_history)

    def validate(self, verbose=True):
        """
        Checks internal consistency of the pAnnData object.
        
        Returns:
            bool: True if all checks pass, False otherwise.
        """
        issues = []

        # --- Check prot and pep dimensions ---
        for label, ad in [('prot', self.prot), ('pep', self.pep)]:
            if ad is not None:
                if ad.obs.shape[0] != ad.X.shape[0]:
                    issues.append(f"{label}.obs rows ({ad.obs.shape[0]}) != {label}.X rows ({ad.X.shape[0]})")
                if ad.var.shape[0] != ad.X.shape[1]:
                    issues.append(f"{label}.var rows ({ad.var.shape[0]}) != {label}.X cols ({ad.X.shape[1]})")
                if ad.obs.index.duplicated().any():
                    issues.append(f"{label}.obs has duplicated index values")
                if ad.var.index.duplicated().any():
                    issues.append(f"{label}.var has duplicated index values")

        # --- Check obs name overlap between prot and pep ---
        if self.prot is not None and self.pep is not None:
            prot_names = set(self.prot.obs_names)
            pep_names = set(self.pep.obs_names)
            if prot_names != pep_names:
                missing_in_pep = prot_names - pep_names
                missing_in_prot = pep_names - prot_names
                issues.append("prot and pep obs_names do not match")
                if missing_in_pep:
                    issues.append(f"  - {len(missing_in_pep)} samples in prot but not in pep")
                if missing_in_prot:
                    issues.append(f"  - {len(missing_in_prot)} samples in pep but not in prot")

        # --- Check .summary alignment ---
        if self._summary is not None:
            for label, ad in [('prot', self.prot), ('pep', self.pep)]:
                if ad is not None:
                    if not ad.obs.index.equals(self._summary.index):
                        issues.append(f"{label}.obs index does not match .summary index")

        # --- Check RS matrix shape + stats ---
        if self.rs is not None and self.prot is not None and self.pep is not None:
            rs_shape = self.rs.shape
            expected_shape = (self.prot.shape[1], self.pep.shape[1])
            if rs_shape != expected_shape:
                issues.append(f"RS shape mismatch: got {rs_shape}, expected {expected_shape} (proteins × peptides)")
            elif verbose:
                nnz = self.rs.nnz if sparse.issparse(self.rs) else np.count_nonzero(self.rs)
                total = self.rs.shape[0] * self.rs.shape[1]
                sparsity = 100 * (1 - nnz / total)
                print(f"ℹ️  RS matrix: {rs_shape} (proteins × peptides), sparsity: {sparsity:.2f}%")

                # Summary stats (works for dense or sparse)
                rs_dense = self.rs.toarray() if sparse.issparse(self.rs) else self.rs
                row_links = rs_dense.sum(axis=1)
                col_links = rs_dense.sum(axis=0)
                print(f"   - Proteins with ≥1 linked peptides: {(row_links > 0).sum()}/{rs_shape[0]}")
                print(f"   - Peptides with ≥1 linked proteins: {(col_links > 0).sum()}/{rs_shape[1]}")
                print(f"   - Mean peptides per protein: {row_links.mean():.2f}")
                print(f"   - Mean proteins per peptide: {col_links.mean():.2f}")

        # --- Summary of results ---
        if issues:
            if verbose:
                print("❌ Validation failed with the following issues:")
                for issue in issues:
                    print(" -", issue)
            return False
        else:
            if verbose:
                print("✅ pAnnData object is valid.")
            return True

    def describe_rs(self):
        """
        Returns a DataFrame summarizing RS (protein × peptide) connectivity:
        - One row per protein
        - Columns: total peptides, unique peptides, etc.
        """
        if self.rs is None:
            print("⚠️ No RS matrix set.")
            return None

        rs = self.rs.toarray() if sparse.issparse(self.rs) else self.rs

        # peptides per protein
        peptides_per_protein = rs.sum(axis=1).astype(int)
        # unique peptides per protein (those mapped only to this protein)
        unique_peptides = (rs.sum(axis=0) == 1)
        unique_counts = rs[:, unique_peptides].sum(axis=1).astype(int)

        summary_df = pd.DataFrame({
            "peptides_per_protein": peptides_per_protein,
            "unique_peptides": unique_counts
        }, index=self.prot.var_names if self.prot is not None else range(rs.shape[0]))

        return summary_df


    def plot_rs(self, figsize=(10, 4)):
        """
        Shows barplots of:
        - Peptides per protein
        - Proteins per peptide
        """
        if self.rs is None:
            print("⚠️ No RS matrix to plot.")
            return

        rs_dense = self.rs.toarray() if sparse.issparse(self.rs) else self.rs
        prot_links = rs_dense.sum(axis=1)
        pep_links = rs_dense.sum(axis=0)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].hist(prot_links, bins=50, color='gray')
        axes[0].set_title("Peptides per Protein")
        axes[0].set_xlabel("Peptide Count")
        axes[0].set_ylabel("Protein Frequency")

        axes[1].hist(pep_links, bins=50, color='gray')
        axes[1].set_title("Proteins per Peptide")
        axes[1].set_xlabel("Protein Count")
        axes[1].set_ylabel("Peptide Frequency")

        plt.tight_layout()
        plt.show()

    def filter_rs(
        self,
        min_peptides_per_protein=None,
        min_unique_peptides_per_protein=2,
        max_proteins_per_peptide=None,
        return_copy=True,
        preset=None,
        validate_after=True
    ):
        """
        Filters the RS matrix and associated .prot and .pep data.

        Parameters:
        - min_peptides_per_protein (int, optional): Keep proteins with ≥ this many total peptides
        - min_unique_peptides_per_protein (int, optional): Keep proteins with ≥ this many unique peptides (default: 2)
        - max_proteins_per_peptide (int, optional): Remove peptides mapped to > this many proteins
        - return_copy (bool): Return a filtered copy if True (default), otherwise modify in place
        - preset (str or dict, optional): Use a predefined filtering strategy:
            * "default" → unique_peptides ≥ 2
            * "lenient" → total peptides ≥ 2
            * dict → custom filter dictionary (same keys as above)
        - validate_after (bool): If True (default), run self.validate() after filtering

        Returns:
        - pAnnData: Filtered copy (if return_copy=True), or None

        Side effects:
        - Adds `.prot.uns['filter_rs']` dictionary with protein/peptide indices kept and summary
        """
        if self.rs is None:
            print("⚠️ No RS matrix to filter.")
            return self if return_copy else None

        # --- Apply preset if given ---
        if preset:
            if preset == "default":
                min_peptides_per_protein = None
                min_unique_peptides_per_protein = 2
                max_proteins_per_peptide = None
            elif preset == "lenient":
                min_peptides_per_protein = 2
                min_unique_peptides_per_protein = None
                max_proteins_per_peptide = None
            elif isinstance(preset, dict):
                min_peptides_per_protein = preset.get("min_peptides_per_protein", min_peptides_per_protein)
                min_unique_peptides_per_protein = preset.get("min_unique_peptides_per_protein", min_unique_peptides_per_protein)
                max_proteins_per_peptide = preset.get("max_proteins_per_peptide", max_proteins_per_peptide)
            else:
                raise ValueError(f"Unknown RS filtering preset: {preset}")

        pdata = self.copy() if return_copy else self

        rs = pdata.rs.toarray() if sparse.issparse(pdata.rs) else pdata.rs

        # --- Step 1: Peptide filter (max proteins per peptide) ---
        if max_proteins_per_peptide is not None:
            peptide_links = rs.sum(axis=0)
            keep_peptides = peptide_links <= max_proteins_per_peptide
            rs = rs[:, keep_peptides]
        else:
            keep_peptides = np.ones(rs.shape[1], dtype=bool)

        # --- Step 2: Protein filters ---
        is_unique = rs.sum(axis=0) == 1
        unique_counts = rs[:, is_unique].sum(axis=1)
        peptide_counts = rs.sum(axis=1)

        keep_proteins = np.ones(rs.shape[0], dtype=bool)
        if min_peptides_per_protein is not None:
            keep_proteins &= (peptide_counts >= min_peptides_per_protein)
        if min_unique_peptides_per_protein is not None:
            keep_proteins &= (unique_counts >= min_unique_peptides_per_protein)

        rs_filtered = rs[keep_proteins, :]

        # --- Step 3: Re-filter peptides now unmapped ---
        keep_peptides_final = rs_filtered.sum(axis=0) > 0
        rs_filtered = rs_filtered[:, keep_peptides_final]

        # --- Apply filtered RS ---
        pdata._set_RS(rs_filtered, validate=False)

        # --- Filter .prot and .pep ---
        if pdata.prot is not None:
            pdata.prot = pdata.prot[:, keep_proteins]
        if pdata.pep is not None:
            original_peptides = keep_peptides.nonzero()[0]
            final_peptides = original_peptides[keep_peptides_final]
            pdata.pep = pdata.pep[:, final_peptides]

        # --- History and summary ---
        n_prot_before = self.prot.shape[1] if self.prot is not None else rs.shape[0]
        n_pep_before = self.pep.shape[1] if self.pep is not None else rs.shape[1]
        n_prot_after = rs_filtered.shape[0]
        n_pep_after = rs_filtered.shape[1]

        n_prot_dropped = n_prot_before - n_prot_after
        n_pep_dropped = n_pep_before - n_pep_after
        
        msg = "🧪 Filtered RS"
        if preset:
            msg += f" using preset '{preset}'"
        if min_peptides_per_protein is not None:
            msg += f", min peptides per protein: {min_peptides_per_protein}"
        if min_unique_peptides_per_protein is not None:
            msg += f", min unique peptides: {min_unique_peptides_per_protein}"
        if max_proteins_per_peptide is not None:
            msg += f", max proteins per peptide: {max_proteins_per_peptide}"
        msg += (
            f". Proteins: {n_prot_before} → {n_prot_after} (dropped {n_prot_dropped}), "
            f"Peptides: {n_pep_before} → {n_pep_after} (dropped {n_pep_dropped})."
        )

        pdata._append_history(msg)
        print(msg)
        pdata.update_summary()

        # --- Save filter indices to .uns ---
        protein_indices = list(pdata.prot.var_names) if pdata.prot is not None else []
        peptide_indices = list(pdata.pep.var_names) if pdata.pep is not None else []
        pdata.prot.uns['filter_rs'] = {
            "kept_proteins": protein_indices,
            "kept_peptides": peptide_indices,
            "n_proteins": len(protein_indices),
            "n_peptides": len(peptide_indices),
            "description": msg
        }

        if validate_after:
            pdata.validate(verbose=True)

        return pdata if return_copy else None

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

    def filter_prot(self, condition = None, accessions=None, return_copy = 'True', debug=False):
        """
        Filters the protein data based on a protein metadata condition or a list of accession numbers.

        Parameters:
        - condition (str): A condition string to filter protein metadata. Can include:
            - Standard comparisons (e.g. `"Protein FDR Confidence: Combined == 'High'"`)
            - Substring search using `includes` (e.g. `"Description includes 'p97'"`)
        - accessions (list of str): List of accession numbers (var_names) to keep.
        - return_copy (bool): If True, returns a filtered copy. If False, modifies in place.
        - debug (bool): If True, prints debugging information.

        Returns:
        - Filtered pAnnData object if `return_copy=True`, else modifies in place and returns None.

        Example:
        >>> condition = "Protein FDR Confidence: Combined == 'High'"
        >>> pdata.filter_prot(condition)
        
        >>> condition = "Description includes 'p97'"
        >>> pdata.filter_prot(condition)

        >>> condition = "Score > 0.75"
        >>> pdata.filter_prot(condition)
        """

        if not self._check_data('protein'):
            raise ValueError(f"No protein data found. Check that protein data was imported.")

        pdata = self.copy() if return_copy else self
        action = "Returning a copy of" if return_copy else "Filtered and modified"

        message_parts = []

        # Filter by condition
        if condition is not None:
            formatted_condition = self._format_filter_query(condition, pdata.prot.var)
            if debug:
                print(f"Formatted condition: {formatted_condition}")
            filtered_proteins = pdata.prot.var[pdata.prot.var.eval(formatted_condition)]
            pdata.prot = pdata.prot[:, filtered_proteins.index]
            message_parts.append(f"condition: {condition} ({pdata.prot.shape[1]} proteins kept)")

        # Filter by accession list
        if accessions is not None:
            existing = pdata.prot.var_names
            present = [acc for acc in accessions if acc in existing]
            missing = [acc for acc in accessions if acc not in existing]

            if missing:
                warnings.warn(f"The following accession(s) were not found and will be ignored: {missing}")

            pdata.prot = pdata.prot[:, pdata.prot.var_names.isin(present)]
            message_parts.append(f"accessions: {len(present)} found / {len(accessions)} requested")

        if not message_parts:
            message = f"{action} protein data. No filters applied."
        else:
            message = f"{action} protein data based on {' and '.join(message_parts)}."

            # TODO: also filter out peptides that belong to the filtered proteins
            # if pdata.pep is not None:
            # need to filter out peptides that belonged only to the filtered proteins, need to use rs matrix for this
            # can start from pdata.prot.var.eval(formatted_condition) to get the rows that we're keeping
            #     pdata.pep = pdata.pep[filtered_queries.index]

        print(message)
        pdata._append_history(message)
        pdata.update_summary(recompute=True)
        return pdata if return_copy else None

    def filter_sample(self, values=None, exact_cases=False, condition=None, file_list=None, return_copy=True, debug=False, query_mode=False):
        """
        Unified method to filter samples in a pAnnData object.

        This function supports three types of filtering input. You must provide **exactly one** of the following:
        
        - `values` (dict or list of dict): For filtering samples based on categorical metadata.
            - Example: `{'treatment': ['kd', 'sc'], 'cellline': 'AS'}`
            - Matches rows in `.summary` or `.obs` with those field values.

        - `condition` (str): A condition string to evaluate against sample-level numeric metadata (summary).
            - Example: `"protein_count > 1000"`
            - This should reference columns in `pdata.summary`.

        - `file_list` (list): A list of sample names or file identifiers to keep.
            - Example: `['Sample_1', 'Sample_2']`
            - Filters to only those samples (must match obs_names).

        Parameters:
        - values (dict or list of dict): Categorical value filter (if used).
        - exact_cases (bool): Use exact combination matching when `values` is provided.
        - condition (str): Summary-level numeric filter condition.
        - file_list (list): List of sample identifiers to keep.
        - return_copy (bool): If True, returns a filtered copy. Otherwise modifies in place.
        - debug (bool): If True, print filter queries/messages.

        Returns:
        - Filtered pAnnData object (if `return_copy=True`), otherwise modifies in place and returns None.

        Raises:
        - ValueError if more than one or none of `values`, `condition`, or `file_list` are specified.

        Examples:
        ---------
        >>> pdata.filter_sample(values={'treatment': 'kd', 'cellline': 'AS'})
        >>> pdata.filter_sample(values=[{'treatment': 'kd', 'cellline': 'AS'}, {'treatment': 'sc', 'cellline': 'BE'}], exact_cases=True)
        >>> pdata.filter_sample(condition="protein_count > 1000")
        >>> pdata.filter_sample(file_list=['Sample_001', 'Sample_007'])
        """

        # Ensure exactly one of the filter modes is specified
        provided = [values, condition, file_list]
        if sum(arg is not None for arg in provided) != 1:
            raise ValueError(
                "Invalid filter input. You must specify exactly one of the following keyword arguments:\n"
                "- `values=...` for categorical metadata filtering,\n"
                "- `condition=...` for summary-level condition filtering, or\n"
                "- `file_list=...` to filter by sample IDs.\n\n"
                "Example:\n"
                "  pdata.filter_sample(condition='protein_quant > 0.2')"
            )

        if values is not None:
            return self.filter_sample_values(
                values=values,
                exact_cases=exact_cases,
                debug=debug,
                return_copy=return_copy
            )

        if condition is not None or file_list is not None:
            return self.filter_sample_metadata(
                condition=condition,
                file_list=file_list,
                return_copy=return_copy,
                debug=debug
            )
        
        if values is not None and query_mode:
            return self.filter_sample_query(query_string=values, source='obs', return_copy=return_copy, debug=debug)

        if condition is not None and query_mode:
            return self.filter_sample_query(query_string=condition, source='summary', return_copy=return_copy, debug=debug)

    def filter_sample_metadata(self, condition = None, return_copy = True, file_list=None, debug=False):
        """
        Filter samples in a pAnnData object based on numeric summary conditions or a file/sample list.

        This method allows filtering using:
        - A string condition referencing columns in `.summary` (e.g. "protein_count > 1000"), or
        - A list of sample identifiers to retain (e.g. filenames or obs_names).

        Parameters:
        - condition (str): A filter condition string evaluated on `pdata.summary`.
        - file_list (list): A list of sample identifiers to retain.
        - return_copy (bool): Return a filtered copy (True) or modify in place (False).
        - debug (bool): Print the query string or summary info.

        Returns:
        - Filtered pAnnData object (or None if in-place).

        Note:
        This method is used internally by `filter_sample()`. For general-purpose filtering, it's recommended
        to use `filter_sample()` and pass either `condition=...` or `file_list=...`.

        Example:
        >>> pdata.filter_sample_metadata("protein_count > 1000")
        or
        >>> pdata.filter_sample_metadata(file_list = ['fileA', 'fileB'])
        """
        if not self._has_data():
            pass

        if self._summary is None:
            self.update_summary(recompute=True)
        
        # Determine whether to operate on a copy or in-place
        pdata = self.copy() if return_copy else self
        action = "Returning a copy of" if return_copy else "Filtered and modified"

        # Define the filtering logic
        if condition is not None:
            formatted_condition = self._format_filter_query(condition, pdata._summary)
            print(formatted_condition) if debug else None
            # filtered_queries = pdata._summary.eval(formatted_condition)
            filtered_samples = pdata._summary[pdata._summary.eval(formatted_condition)]
            index_filter = filtered_samples.index
            message = f"{action} data based on sample condition: {condition}. Number of samples kept: {len(filtered_samples)}."
            # Number of samples dropped: {len(pdata._summary) - len(filtered_samples)}
        elif file_list is not None:
            index_filter = file_list
            message = f"{action} data based on sample file list. Number of samples kept: {len(file_list)}."
            missing = [f for f in file_list if f not in pdata.prot.obs_names]
            if missing:
                warnings.warn(f"Some sample IDs not found: {missing}")
            # Number of samples dropped: {len(pdata._summary) - len(file_list)}
        else:
            # No filtering applied
            message = "No filtering applied. Returning original data."
            return pdata if return_copy else None

        # Filter out selected samples from prot and pep
        if pdata.prot is not None:
            pdata.prot = pdata.prot[pdata.prot.obs_names.isin(index_filter)]
        
        if pdata.pep is not None:
            pdata.pep = pdata.pep[pdata.pep.obs_names.isin(index_filter)]

        # Logging and history updates
        print(message)
        pdata._append_history(message)
        pdata.update_summary(recompute=False)

        return pdata if return_copy else None
    
    def filter_sample_values(self, values, exact_cases, debug=False, return_copy=True):
        """
        Filter samples in a pAnnData object using dictionary-style categorical matching.

        This method allows filtering based on class-like annotations (e.g. treatment, cellline)
        using either simple OR logic within fields, or exact AND combinations across fields.

        Parameters:
        - pdata (AnnData): Input AnnData object
        - values (dict or list of dict): Filtering conditions
            - If `exact_cases=False`: dictionary of field: values (OR within field, AND across fields)
            - If `exact_cases=True`: list of dictionaries, each defining an exact match case
        - exact_cases (bool): Enable exact combination matching
        - debug (bool): Suppress query printing
        - return_copy (bool): Return a copy of the filtered object

        Returns:
        - AnnData: Filtered view of input data

        Note:
        This method is used internally by `filter_sample()`. For most use cases, calling `filter_sample()`
        directly is preferred.
        
        Examples:
        ---------
        >>> pdata.filter_sample_values(values={'treatment': ['kd', 'sc'], 'cellline': 'AS'})
        # History: Filtered by loose match on: treatment: ['kd', 'sc'], cellline: AS. Number of samples kept: 42. Copy of the filtered AnnData object returned.

        >>> pdata.filter_sample_values(pdatvalues=[
        ...     {'treatment': 'kd', 'cellline': 'AS'},
        ...     {'treatment': 'sc', 'cellline': 'BE'}
        ... ], exact_cases=True)
        # History: Filtered by exact match on: {'treatment': 'kd', 'cellline': 'AS'}; {'treatment': 'sc', 'cellline': 'BE'}. Number of samples kept: 17.
        """

        pdata = self.copy() if return_copy else self
        obs_keys = pdata.summary.columns

        if exact_cases:
            if not isinstance(values, list) or not all(isinstance(v, dict) for v in values):
                raise ValueError("When exact_cases=True, `values` must be a list of dictionaries.")

            for case in values:
                if not case:
                    raise ValueError("Empty dictionary found in values.")
                for key in case:
                    if key not in obs_keys:
                        raise ValueError(f"Field '{key}' not found in adata.obs.")

            query = " | ".join([
                " & ".join([
                    f"(adata.obs['{k}'] == '{v}')" for k, v in case.items()
                ])
                for case in values
            ])

        else:
            if not isinstance(values, dict):
                raise ValueError("When exact_cases=False, `values` must be a dictionary.")

            for key in values:
                if key not in obs_keys:
                    raise ValueError(f"Field '{key}' not found in adata.obs.")

            query_parts = []
            for k, v in values.items():
                v_list = v if isinstance(v, list) else [v]
                part = " | ".join([f"(adata.obs['{k}'] == '{val}')" for val in v_list])
                query_parts.append(f"({part})")
            query = " & ".join(query_parts)

        if debug:
                print(f"Filter query: {query}")

        if pdata.prot is not None:
            adata = pdata.prot
            pdata.prot = adata[eval(query)]
        if pdata.pep is not None:
            adata = pdata.pep
            pdata.pep = adata[eval(query)]

        n_samples = len(pdata.prot)
        filter_desc = (
            f"Filtered by exact match on: {'; '.join(map(str, values))}."
            if exact_cases else
            f"Filtered by loose match on: {', '.join(f'{k}: {v}' for k, v in values.items())}."
        )
        copy_note = " Copy of the filtered AnnData object returned." if return_copy else "Filtered and modified in-place."
        history_msg = f"{filter_desc} Number of samples kept: {n_samples}.{copy_note}"
        pdata._append_history(history_msg)  
        print(history_msg)
        if n_samples == 0:
            print("Warning: No samples matched the filter criteria.")
        pdata.update_summary(recompute=False)

        return pdata

    def filter_sample_query(self, query_string, source='obs', return_copy=True, debug=False):
        """
        Filters samples using a raw pandas-style query string on either obs or summary.

        Parameters:
        - query_string (str): A pandas-style query string (e.g., "cellline == 'AS' and treatment in ['kd', 'sc']")
        - source (str): Either 'obs' or 'summary' — the dataframe to evaluate the query against.
        - return_copy (bool): Return a new filtered object or modify in place.
        - debug (bool): Print debug messages.

        Returns:
        - Filtered pAnnData object or modifies in place.
        """
        pdata = self.copy() if return_copy else self
        action = "Returning a copy of" if return_copy else "Filtered and modified"

        print("⚠️  Advanced query mode enabled — interpreting string as a pandas-style expression.")

        if source == 'obs':
            df = pdata.prot.obs
        elif source == 'summary':
            if self._summary is None:
                self.update_summary(recompute=True)
            df = pdata._summary
        else:
            raise ValueError("source must be 'obs' or 'summary'")

        try:
            filtered_df = df.query(query_string)
        except Exception as e:
            raise ValueError(f"Failed to parse query string:\n  {query_string}\nError: {e}")

        index_filter = filtered_df.index

        if pdata.prot is not None:
            pdata.prot = pdata.prot[pdata.prot.obs_names.isin(index_filter)]
        if pdata.pep is not None:
            pdata.pep = pdata.pep[pdata.pep.obs_names.isin(index_filter)]

        message = f"{action} samples based on query string. Samples kept: {len(index_filter)}."
        print(message)
        pdata._append_history(message)
        pdata._update_summary(recompute=False)

        return pdata if return_copy else None


    def export(self, filename, format = 'csv'):
        # export data, each layer as a separate file
        
        # if filename not specified, use current date and time
        if filename is None:
            filename = setup.get_datetime()

        if not self._has_data():
            raise ValueError("No data found in pAnnData object.")
        
        # export summary
        self._summary.to_csv(f"{filename}_summary.csv")

        if self.prot is not None:
            self.prot.to_df().to_csv(f"{filename}_protein.csv")
            for layer in self.prot.layers:
                self.prot.layers[layer].toarray().to_csv(f"{filename}_protein_{layer}.csv")

    def _format_filter_query(self, condition, dataframe):
        """
        Formats a query string for filtering a DataFrame with potentially complex column names. Used in `filter_sample_metadata()` and `filter_prot()`.

        - Wraps column names containing spaces/special characters in backticks for `pandas.eval()`.
        - Supports custom `includes` syntax for substring matching, e.g.:
            "Description includes 'p97'" → `Description.str.contains('p97', case=False, na=False)`
        - Auto-quotes unquoted string values when column dtype is object or category.

        Parameters:
        - condition (str): The condition string to parse.
        - dataframe (pd.DataFrame): DataFrame whose columns will be used for parsing.

        Returns:
        - str: A condition string formatted for pandas `.eval()`
        """

        # Wrap column names with backticks if needed
        column_names = dataframe.columns.tolist()
        column_names.sort(key=len, reverse=True) # Avoid partial matches
        
        for col in column_names:
            if re.search(r'[^\\w]', col):  # Non-alphanumeric characters
                condition = re.sub(fr'(?<!`)({re.escape(col)})(?!`)', f'`{col}`', condition)

        # Handle 'includes' syntax for substring matching
        match = re.search(r'`?(\w[\w\s:.-]*)`?\s+includes\s+[\'"]([^\'"]+)[\'"]', condition)
        if match:
            col_name = match.group(1)
            substring = match.group(2)
            condition = f"{col_name}.str.contains('{substring}', case=False, na=False)"

        # Auto-quote string values for categorical/text columns
        for col in dataframe.columns:
            if dataframe[col].dtype.name in ["object", "category"]:
                for op in ["==", "!="]:
                    pattern = fr"(?<![><=!])\b{re.escape(col)}\s*{op}\s*([^\s'\"()]+)"
                    matches = re.findall(pattern, condition)
                    for match_val in matches:
                        quoted_val = f'"{match_val}"'
                        condition = re.sub(fr"({re.escape(col)}\s*{op}\s*){match_val}\b", r"\1" + quoted_val, condition)

        return condition

    # -----------------------------
    # PROCESSING FUNCTIONS
    def cv(self, classes = None, on = 'protein', layer = "X", debug = False):
        if not self._check_data(on):
            pass

        adata = self.prot if on == 'protein' else self.pep
        classes_list = utils.get_classlist(adata, classes)

        for j, class_value in enumerate(classes_list):
            data_filtered = utils.resolve_class_filter(adata, classes, class_value, debug=True)

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
            rank_data = utils.resolve_class_filter(adata, classes, class_value, debug=True)

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

    # TODO: imputation between samples - need to figure out how to split the data into classes before running imputation, and then splicing back together
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

# TODO: move to class function
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

def import_data(source: str, **kwargs):
    """
    Unified wrapper for importing data into a pAnnData object.
    
    Parameters:
    - source (str): The tool or data source. Options:
        - 'diann' or 'dia-nn' → calls import_diann()
        - 'pd', 'proteomeDiscoverer', 'pd13', 'pd24' → calls import_proteomeDiscoverer()
        - 'fragpipe', 'fp' → Not implemented yet
        - 'spectronaut', 'sn' → Not implemented yet
    - **kwargs: Arguments passed directly to the corresponding import function

    Returns:
    - pAnnData object
    """
    source = source.lower()

    if source in ['diann', 'dia-nn']:
        return import_diann(**kwargs)

    elif source in ['pd', 'proteomediscoverer', 'proteome_discoverer', 'pd2.5', 'pd24']:
        return import_proteomeDiscoverer(**kwargs)

    elif source in ['fragpipe', 'fp']:
        raise NotImplementedError("FragPipe import is not yet implemented. Stay tuned!")

    elif source in ['spectronaut', 'sn']:
        raise NotImplementedError("Spectronaut import is not yet implemented. Stay tuned!")

    else:
        raise ValueError(f"❌ Unsupported import source: '{source}'. "
                         "Valid options: 'diann', 'proteomeDiscoverer', 'fragpipe', 'spectronaut'.")

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
        pep_X = pep_layers_mbr = pep_var_names = pep_var = pep_obs_names = pep_obs = None

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
    # check if mlb.classes_ has overlap with prot_var
    if prot_file and pep_file:
        mlb_classes_set = set(mlb.classes_)
        prot_var_set = set(prot_var_names)

        if mlb_classes_set != prot_var_set:
            print("WARNING: Master proteins in the peptide matrix do not match proteins in the protein data, please check if files correspond to the same data.")
            print(f"Overlap: {len(mlb_classes_set & prot_var_set)}")
            print(f"Unique to peptide data: {mlb_classes_set - prot_var_set}")
            print(f"Unique to protein data: {prot_var_set - mlb_classes_set}")

    pdata = _create_pAnnData_from_parts(
        prot_X, pep_X, rs,
        prot_obs, prot_var, prot_obs_names, prot_var_names,
        pep_obs, pep_var, pep_obs_names, pep_var_names,
        obs_columns=obs_columns,
        X_mbr_prot=prot_layers_mbr,
        X_mbr_pep=pep_layers_mbr,
        metadata={
            "source": "proteomeDiscoverer",
            "prot_file": prot_file,
            "pep_file": pep_file
        },
        history_msg="Imported protein and/or peptide data from Proteome Discoverer."
    )

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
            # check if 'PG.Quantity' is in the columns, if yes then pass, if not then throw an error that DIA-NN version >2.0 does not have PG.quantity
            if 'PG.Quantity' not in report_all.columns:
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
    # print("RS matrix successfully computed")

    # TODO: number of peptides per protein
    # number of proteins per peptide
    # make columns isUnique for peptide
    sum_values = np.sum(rs, axis=1)

    # -----------------------------
    # ASSERTIONS
    # -----------------------------
    # check if mlb.classes_ has overlap with prot_var
    mlb_classes_set = set(mlb.classes_)
    prot_var_set = set(prot_var_names)

    if mlb_classes_set != prot_var_set:
        print("WARNING: Master proteins in the peptide matrix do not match proteins in the protein data, please check if files correspond to the same data.")
        print(f"Overlap: {len(mlb_classes_set & prot_var_set)}")
        print(f"Unique to peptide data: {mlb_classes_set - prot_var_set}")
        print(f"Unique to protein data: {prot_var_set - mlb_classes_set}")
    
    pdata = _create_pAnnData_from_parts(
        prot_X, pep_X, rs,
        prot_obs, prot_var, prot_obs_names, prot_var_names,
        pep_obs, pep_var, pep_obs_names, pep_var_names,
        obs_columns=obs_columns,
        metadata={
            "source": "diann",
            "file": report_file,
            "protein_metric": prot_value,
            "peptide_metric": pep_value
        },
        history_msg=f"Imported DIA-NN report from {report_file} using {prot_value} (protein) and {pep_value} (peptide)."
    )
    
    pdata._append_history(f"Imported protein data from {report_file}, using {prot_value} as protein value and {pep_value} as peptide value.")
    print("pAnnData object created. Use `print(pdata)` to view the object.")

    return pdata

def _create_pAnnData_from_parts(
    prot_X, pep_X, rs,
    prot_obs, prot_var, prot_obs_names, prot_var_names,
    pep_obs=None, pep_var=None, pep_obs_names=None, pep_var_names=None,
    obs_columns=None,
    X_mbr_prot=None,
    X_mbr_pep=None,
    metadata=None,
    history_msg=""
):
    """
    Assemble a pAnnData object from processed matrices and metadata.

    Parameters:
    - prot_X, pep_X: csr_matrix (observations × features); one may be None
    - rs: peptide-to-protein relational matrix (or None if not applicable)
    - *_obs, *_var: pandas DataFrames for sample and feature metadata
    - *_obs_names, *_var_names: list-like of sample/protein/peptide names
    - obs_columns: optional list of column names to assign to .obs
    - X_mbr_prot, X_mbr_pep: optional MBR identification layers
    - metadata: optional dict of metadata tags (e.g. {'source': 'diann'})
    - history_msg: string to append to the object's history

    Returns:
    - pAnnData object with summary updated and validated
    """
    pdata = pAnnData(prot_X, pep_X, rs)

    # --- PROTEIN ---
    if prot_X is not None:
        pdata.prot.obs = pd.DataFrame(prot_obs)
        pdata.prot.var = pd.DataFrame(prot_var)
        pdata.prot.obs_names = list(prot_obs_names)
        pdata.prot.var_names = list(prot_var_names)
        pdata.prot.obs.columns = obs_columns if obs_columns else list(range(pdata.prot.obs.shape[1]))
        pdata.prot.layers['X_raw'] = prot_X
        if X_mbr_prot is not None:
            pdata.prot.layers['X_mbr'] = X_mbr_prot

    # --- PEPTIDE ---
    if pep_X is not None:
        pdata.pep.obs = pd.DataFrame(pep_obs)
        pdata.pep.var = pd.DataFrame(pep_var)
        pdata.pep.obs_names = list(pep_obs_names)
        pdata.pep.var_names = list(pep_var_names)
        pdata.pep.obs.columns = obs_columns if obs_columns else list(range(pdata.pep.obs.shape[1]))
        pdata.pep.layers['X_raw'] = pep_X
        if X_mbr_pep is not None:
            pdata.pep.layers['X_mbr'] = X_mbr_pep

    # --- Metadata ---
    metadata = metadata or {}
    metadata.setdefault("imported_at", datetime.datetime.now().isoformat())

    if pdata.prot is not None:
        pdata.prot.uns['metadata'] = metadata
    if pdata.pep is not None:
        pdata.pep.uns['metadata'] = metadata

    # --- Summary + Validation ---
    pdata.update_summary(recompute=True)

    if not pdata.validate():
        print("⚠️ Warning: Validation issues found. Use `pdata.validate()` to inspect.")

    if history_msg:
        pdata._append_history(history_msg)

    return pdata
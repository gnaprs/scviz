import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MetricsMixin:
    """
    Computes descriptive and RS-derived metrics for proteomic data.

    This mixin provides utility functions for calculating summary statistics on
    protein and peptide abundance data, as well as inspecting the structure of
    the RS (protein × peptide) relational matrix.

    Features:

    - Computes per-sample quantification and abundance metrics for both proteins and peptides
    - Calculates RS-derived properties such as the number of peptides per protein and the number of unique peptides
    - Updates `.obs`, `.var`, and `.summary` with relevant metrics
    - Provides visualization and tabular summaries of RS matrix connectivity

    Functions:
        _update_metrics: Computes per-sample and RS-derived metrics for `.prot` and `.pep`.
        _update_summary_metrics: Adds per-sample high-confidence protein counts to `.summary`.
        describe_rs: Returns a DataFrame summarizing peptide connectivity per protein.
        plot_rs: Generates histograms of peptide–protein and protein–peptide mapping counts.
    """
    def _update_metrics(self):
        """
        Compute and update core QC and RS-based metrics for `.obs` and `.var`.

        This internal method updates:
        
        - `.prot.obs` and `.pep.obs` with per-sample metrics:
            • `*_quant`: Proportion of non-missing values
            • `*_count`: Number of non-missing values
            • `*_abundance_sum`: Sum of observed abundances
            • `mbr_count`, `high_count`: Count of MBR annotations (if present in layer 'X_mbr')
        
        - `.prot.var` with RS-derived metrics (if available):
            • `peptides_per_protein`: Total peptides mapped to each protein
            • `unique_peptides`: Number of peptides uniquely mapping to each protein

        Note:
            This function is typically called automatically after filtering, imputation,
            or importing new data. It should not be run manually under normal usage.
        """
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
            rs = self.rs  # leave it sparse
            peptides_per_protein = rs.getnnz(axis=1)
            unique_mask = rs.getnnz(axis=0) == 1
            unique_counts = rs[:, unique_mask].getnnz(axis=1)
            self.prot.var['peptides_per_protein'] = peptides_per_protein
            self.prot.var['unique_peptides'] = unique_counts

    def _update_summary_metrics(self, unique_peptide_thresh=2):
        """
        Compute RS-derived per-sample summary metric for protein confidence.

        Adds the following column to `.summary`:
        
        - `unique_pep2_protein_count`: Number of proteins per sample with at least 
        `unique_peptide_thresh` uniquely mapping peptides (default: 2).

        This is useful for quality control and filtering based on protein-level confidence.

        Args:
            unique_peptide_thresh (int): Minimum number of uniquely mapping peptides required
                to consider a protein as confidently quantified.
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

    def describe_rs(self):
        """
        Summarize the protein–peptide RS (relational) matrix.

        Returns a DataFrame with one row per protein, describing its peptide mapping coverage:

        - `peptides_per_protein`: Total number of peptides mapped to each protein.
        - `unique_peptides`: Number of uniquely mapping peptides (peptides linked to only one protein).

        Returns:
            pd.DataFrame: Summary statistics for each protein in the RS matrix.

        Note:
            If `.prot` is available, index labels are taken from `.prot.var_names`.
        """
        if self.rs is None:
            print("⚠️ No RS matrix set.")
            return None

        rs = self.rs

        # peptides per protein
        peptides_per_protein = rs.getnnz(axis=1)
        # unique peptides per protein (those mapped only to this protein)
        unique_mask = rs.getnnz(axis=0) == 1
        unique_counts = rs[:, unique_mask].getnnz(axis=1)

        summary_df = pd.DataFrame({
            "peptides_per_protein": peptides_per_protein,
            "unique_peptides": unique_counts
        }, index=self.prot.var_names if self.prot is not None else range(rs.shape[0]))

        return summary_df


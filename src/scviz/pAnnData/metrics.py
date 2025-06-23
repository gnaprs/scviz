import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MetricsMixin:
    """
    Handles descriptive and relational statistics.

    Functions:
        _update_metrics: Computes RS-derived metrics and updates `.var`.
        describe_rs: Summarizes RS matrix structure.
        plot_rs: Visualizes RS matrix statistics across proteins/peptides.
    """
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
            rs = self.rs  # leave it sparse
            peptides_per_protein = rs.getnnz(axis=1)
            unique_mask = rs.getnnz(axis=0) == 1
            unique_counts = rs[:, unique_mask].getnnz(axis=1)
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

    def describe_rs(self):
        """
        Returns a DataFrame summarizing RS (protein × peptide) connectivity:
        - One row per protein
        - Columns: total peptides, unique peptides, etc.
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

    def plot_rs(self, figsize=(10, 4)):
        """
        Shows barplots of:
        - Peptides per protein
        - Proteins per peptide
        """
        if self.rs is None:
            print("⚠️ No RS matrix to plot.")
            return

        rs = self.rs
        prot_links = rs.getnnz(axis=1)
        pep_links = rs.getnnz(axis=0)

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


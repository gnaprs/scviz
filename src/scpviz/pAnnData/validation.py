from scpviz.utils import format_log_prefix
from scipy import sparse
import numpy as np

class ValidationMixin:
    """
    Provides internal validation checks for data consistency across `.prot`, `.pep`, `.summary`, and the RS matrix.

    This mixin ensures that core components of the pAnnData object are structurally aligned and reports any mismatches
    or inconsistencies in dimensionality, identifiers, and matrix shapes.

    Functions:

        validate:
            Comprehensive integrity check for .prot, .pep, .summary, and RS matrix. Returns True if all checks pass.

        _check_data:
            Utility method to verify that protein or peptide data exists. Raises error if missing.

        _check_rankcol:
            Verifies presence of 'Average: <class>' and 'Rank: <class>' columns in `.var` for use with rank-based plotting.
    """
    def validate(self, verbose=True):
        """
        Check internal consistency of the pAnnData object.

        This function verifies that `.prot`, `.pep`, `.summary`, and `.rs` are 
        internally aligned, with matching dimensions, index values, and consistency
        between shared sample identifiers. It prints helpful diagnostics if issues
        are detected.

        Checks Performed:
        -----------------
        - `.obs` and `.var` shapes match `.X` for both `.prot` and `.pep`
        - `.obs` and `.var` indices are unique
        - `.prot.obs_names` match `.pep.obs_names`
        - `.summary.index` matches `.obs.index` for both `.prot` and `.pep`
        - `.rs.shape` matches (n_proteins, n_peptides)
        - Prints RS matrix sparsity and connectivity stats (if verbose=True)

        Args:
            verbose (bool): If True, print summary of validation and RS stats.

        Returns:
            bool: True if all checks pass, False otherwise.

        Example:
            To validate the pAnnData object and check for consistency issues:
                ```python
                is_valid = pdata.validate()
                if not is_valid:
                    print("Fix issues before proceeding.")
                ```
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
                print(f"{format_log_prefix('info_only', indent=1)} RS matrix: {rs_shape} (proteins × peptides), sparsity: {sparsity:.2f}%")

                rs = self.rs

                row_links = rs.getnnz(axis=1)  # peptides per protein
                col_links = rs.getnnz(axis=0)  # proteins per peptide

                # Unique peptides (linked to only 1 protein)
                unique_peptides_mask = col_links == 1
                unique_counts = rs[:, unique_peptides_mask].getnnz(axis=1)  # unique peptides per protein

                # Summary stats
                print(f"   - Proteins with ≥2 *unique* linked peptides: {(unique_counts >= 2).sum()}/{rs_shape[0]}")
                print(f"   - Peptides linked to ≥2 proteins: {(col_links >= 2).sum()}/{rs_shape[1]}")
                print(f"   - Mean peptides per protein: {row_links.mean():.2f}")
                print(f"   - Mean proteins per peptide: {col_links.mean():.2f}")

        # --- Summary of results ---
        if issues:
            if verbose:
                print(f"{format_log_prefix('error')} Validation failed with the following issues:")
                for issue in issues:
                    print(" -", issue)
            return False
        else:
            if verbose:
                print(f"{format_log_prefix('result')} pAnnData object is valid.")
            return True

    def _check_data(self, on):
        """
        Internal check for existence of protein or peptide data.

        Args:
            on (str): One of 'protein', 'peptide', 'prot', or 'pep'.

        Returns:
            bool: True if corresponding data exists.

        Raises:
            ValueError: If `on` is not a valid option or if the corresponding data is missing.
        """
        # check if protein or peptide data exists
        if on not in ['protein', 'peptide' , 'prot', 'pep']:
            raise ValueError("Invalid input: on must be either 'protein' or 'peptide'.")
        elif (on == 'protein' or on == 'prot') and self.prot is None:
            raise ValueError("No protein data found in AnnData object.")
        elif (on == 'peptide' or on == 'pep') and self.pep is None:
            raise ValueError("No peptide data found in AnnData object.")
        else:
            return True

    def _check_rankcol(self, on = 'protein', class_values = None):
        """
        Internal check for existence of average and rank columns in `.var`.

        This function ensures that for each `class_value`, both of the following 
        columns exist in the `.var` of the chosen modality:
            - 'Average: <class_value>'
            - 'Rank: <class_value>'

        Args:
            on (str): 'protein' or 'peptide'.
            class_values (list of str): Class values expected to have been used in plot_rankquank().

        Raises:
            ValueError: If class_values is None, or if required columns are missing from `.var`.
        """
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


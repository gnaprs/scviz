import numpy as np
import pandas as pd
import warnings
from scipy import sparse
from scviz import utils
from scviz.utils import format_log_prefix

class EditingMixin:
    """
    Provides utilities for modifying core components of a `pAnnData` object, including 
    matrix layers, abundance formatting, exports, and the protein–peptide mapping.

    This mixin includes utilities for:

    - Replacing `.X` with a specific layer from protein or peptide data.
    - Extracting long-form abundance DataFrames with metadata for plotting or analysis.
    - Exporting internal data (summary, matrix layers) to disk.
    - Setting or updating the RS (protein × peptide) relational mapping matrix.

    Functions:
        set_X: Sets the `.X` matrix of protein or peptide data to a specified layer.
        get_abundance: Returns long-form abundance + metadata for selected features.
        export: Exports summary, matrix values, and layers to CSV.
        _set_RS: Sets the RS (protein × peptide) mapping matrix, with optional validation.
    """

    def set_X(self, layer, on = 'protein'):
        """
        Set the `.X` matrix of protein or peptide data to a specified layer.

        This method replaces the active `.X` matrix with the contents of a named layer 
        from `.prot.layers` or `.pep.layers`. This is useful for switching between 
        different processing stages (e.g., normalized, imputed, or raw data).

        Args:
            layer (str): Name of the data layer to assign to `.X`.
            on (str): Whether to operate on `"protein"` or `"peptide"` data (default is `"protein"`).

        Returns:
            None

        Example:
            Set the protein matrix `.X` to the "normalized" layer:
                ```python
                pdata.set_X(layer="normalized", on="protein")
                ```

            Set the peptide matrix `.X` to the "imputed" layer:
                ```python
                pdata.set_X(layer="imputed", on="peptide")
                ```
        """
        # defines which layer to set X to
        if not self._check_data(on): # type: ignore[attr-defined]
            pass

        if on == 'protein':
            if layer not in self.prot.layers: # type: ignore[attr-defined]
                raise ValueError(f"Layer {layer} not found in protein data.")
            self.prot.X = self.prot.layers[layer] # type: ignore[attr-defined]
            print(f"{format_log_prefix('info_only', indent=2)} Set {on} data to layer {layer}.")

        else:
            if layer not in self.pep.layers: # type: ignore[attr-defined]
                raise ValueError(f"Layer {layer} not found in peptide data.")
            self.pep.X = self.pep.layers[layer] # type: ignore[attr-defined]
            print(f"{format_log_prefix('info_only', indent=2)} Set {on} data (.X) to layer {layer}.")

        self._history.append(f"{on}: Set X to layer {layer}.") # type: ignore[attr-defined]

    def get_abundance(self, namelist=None, layer='X', on='protein', classes=None, log=True, x_label='gene'):
        """
        Extract a long-form abundance DataFrame from a pAnnData object.

        This method returns a melted (long-form) DataFrame containing abundance values
        along with optional sample metadata and protein/peptide annotations.

        Args:
            namelist (list of str, optional): List of accessions or gene names to extract. If None, returns all features.
            layer (str): Name of the data layer to use (default is "X").
            on (str): Whether to extract from "protein" or "peptide" data.
            classes (str or list of str, optional): Sample-level `.obs` column(s) to include for grouping or plotting.
            log (bool): If True, applies log2 transform to abundance values.
            x_label (str): Whether to label features by "gene" or "accession" in the output.

        Returns:
            pd.DataFrame: Long-form DataFrame with abundance values and associated metadata.

        Example:
            Extract abundance values for selected proteins, grouped by sample-level metadata:
                ```python
                df_abund = pdata.get_abundance(
                    namelist=["UBE4B", "GAPDH"],
                    on="protein",
                    classes=["treatment", "cellline"]
                )
                ```

        Note:
            This method is also available as a utility function in utils, for `AnnData` or `pAnnData` objects:
                ```python
                from scutils import get_abundance
                df_abund = get_abundance(pdata, namelist=["UBE4B", "GAPDH"], on="protein", classes=["treatment", "cellline"])
                ```
        """
        on_user = on.lower()
        gene_to_acc, _ = self.get_gene_maps(on='protein' if on_user in ('peptide', 'pep') else on_user) # type: ignore[attr-defined]


        if on == 'peptide' and namelist:
            pep_names = self.pep.var_names.astype(str) # type: ignore[attr-defined]
            matched_peptides = [name for name in namelist if name in pep_names]
            non_peptides = [name for name in namelist if name not in matched_peptides]

            adata = None
            if len(matched_peptides) < len(namelist):
                filtered = self.filter_prot(accessions=non_peptides, return_copy=True) # type: ignore[attr-defined]
                adata = filtered.pep

            if matched_peptides:
                direct_peps = self.pep[:, matched_peptides] # type: ignore[attr-defined]
                adata = direct_peps if adata is None else adata.concatenate(direct_peps, join='outer')

            if adata is None or adata.n_vars == 0:
                raise ValueError("No matching peptides found from the provided `namelist`.")

            adata = adata[:, ~adata.var_names.duplicated()]

        else:
            adata = utils.get_adata(self, on)

            if namelist:
                resolved = utils.resolve_accessions(adata, namelist, gene_map=gene_to_acc)
                adata = adata[:, resolved]

        # Extract the abundance matrix
        X = adata.layers[layer] if layer in adata.layers else adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Melt into long form
        df = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names).reset_index()
        df = df.melt(id_vars="index", var_name="accession", value_name="abundance")
        df = df.rename(columns={"index": "cell"})

        # Merge obs metadata
        df = df.merge(adata.obs.reset_index(), left_on="cell", right_on="index")

        _, pep_to_prot = self.get_gene_maps(on='peptide')  # peptide → protein map, # type: ignore[attr-defined]
        _, acc_to_gene = self.get_gene_maps(on='protein')  # protein accession → gene, # type: ignore[attr-defined]
        # Map to gene names
        if on == 'peptide':
            try:
                df['protein_accession'] = df['accession'].map(pep_to_prot)
                df['gene'] = df['protein_accession'].map(acc_to_gene)

                # Report unmapped peptides
                unmapped = df[df['gene'].isna()]['accession'].unique().tolist()
                if unmapped:
                    print(f"[get_abundance] {len(unmapped)} peptides could not be mapped to genes: {unmapped}")
            except Exception as e:
                print(f"[get_abundance] Mapping error: {e}")
                df['gene'] = None
        else:
            df['gene'] = df['accession'].map(acc_to_gene)
        
        # Determine x-axis label
        if x_label == 'gene':
            df['x_label_name'] = df['gene'].fillna(df['accession'])
        elif x_label == 'accession':
            if on == 'protein':
                df['x_label_name'] = df['accession']
            elif on == 'peptide':
                try:
                    mapping_col = utils.get_pep_prot_mapping(self)
                    pep_to_prot = self.pep.var[mapping_col].to_dict() # type: ignore[attr-defined]
                    df['x_label_name'] = df['protein_accession']
                except Exception as e:
                    warnings.warn(f"Could not map peptides to accessions: {e}")
                    df['x_label_name'] = df['accession']
        else:
            df['x_label_name'] = df['accession']  # fallback

        # Annotate class/grouping
        if classes:
            df['class'] = df[classes] if isinstance(classes, str) else df[classes].astype(str).agg('_'.join, axis=1)
        else:
            df['class'] = 'all'

        # Log transform
        if log:
            df['log2_abundance'] = np.log2(np.clip(df['abundance'], 1e-6, None))

        return df

    def export(self, filename, format = 'csv', verbose = True):
        """
        Export the pAnnData object's contents to file, including layers and summary metadata.

        This method saves the summary table, protein matrix, and all data layers as separate 
        CSV files using the specified filename as a prefix.

        Args:
            filename (str): Prefix for exported files. If None, uses the current date and time.
            format (str): File format to export (default is "csv").
            verbose (bool): Whether to print progress messages.

        Returns:
            None

        Todo:
            Add example usage showing how to export data and where files are saved. (HDF5, Parquet?)
        """
        if filename is None:
            filename = setup.get_datetime()

        if not self._has_data():
            raise ValueError("No data found in pAnnData object.")
        
        if verbose:
            print(f"{format_log_prefix('user')} Exporting pAnnData to <{filename}>...")
        
        # --- Summary ---
        self.summary.to_csv(f"{filename}_summary.csv")
        if verbose:
            print(f"{format_log_prefix('result_only',2)} Exported summary table → {filename}_summary.csv")

        # --- Protein matrix ---
        if self.prot is not None:
            self.prot.to_df().to_csv(f"{filename}_protein.csv")
            if verbose:
                print(f"{format_log_prefix('result_only',2)} Exported protein matrix → {filename}_protein.csv")

            for layer in self.prot.layers:
                arr = self.prot.layers[layer]
                if hasattr(arr, 'toarray'):
                    arr = arr.toarray()
                df = pd.DataFrame(arr, index=self.prot.obs_names, columns=self.prot.var_names)
                df.to_csv(f"{filename}_protein_{layer}.csv")
                if verbose:
                    print(f"{format_log_prefix('result_only',2)} Exported protein layer '{layer}' → {filename}_protein_{layer}.csv")

        # --- Peptide matrix ---
        if self.pep is not None:
            self.pep.to_df().to_csv(f"{filename}_peptide.csv")
            if verbose:
                print(f"{format_log_prefix('result_only',2)} Exported peptide matrix → {filename}_peptide.csv")

            for layer in self.pep.layers:
                arr = self.pep.layers[layer]
                if hasattr(arr, 'toarray'):
                    arr = arr.toarray()
                df = pd.DataFrame(arr, index=self.pep.obs_names, columns=self.pep.var_names)
                df.to_csv(f"{filename}_peptide_{layer}.csv")
                if verbose:
                    print(f"{format_log_prefix('result_only',2)} Exported peptide layer '{layer}' → {filename}_peptide_{layer}.csv")

    def export_layer(self, layer_name, filename=None, on='protein', obs_names=None, var_names=None, transpose=False):
        """
        Export a specified layer from the protein or peptide data to CSV with labeled rows and columns.

        Args:
            layer_name (str): Name of the layer to export (e.g., "X_raw").
            filename (str, optional): Output file name. Defaults to "<layer_name>.csv".
            on (str): One of 'protein' or 'peptide' to specify which data to use.
            obs_names (str or None): If a string, the column name in .obs to use for row labels.
            var_names (str or None): If a string, the column name in .var to use for column labels.
            transpose: If True, then export as proteins/peptides (rows) by samples (columns)

        Returns:
            None
        """
        # Select the appropriate AnnData object
        adata = self.prot if on == 'protein' else self.pep
        layer = adata.layers[layer_name]

        # Convert to dense array if needed
        if not isinstance(layer, pd.DataFrame):
            layer = layer.toarray() if hasattr(layer, 'toarray') else layer

        # Get row (obs) and column (var) labels
        row_labels = adata.obs[obs_names] if obs_names else adata.obs_names
        col_labels = adata.var[var_names] if var_names else adata.var_names

        # Build the DataFrame
        if transpose:
            df = pd.DataFrame(layer.T, columns=row_labels, index=col_labels)
        else:
            df = pd.DataFrame(layer, index=row_labels, columns=col_labels)

        # Save to CSV
        if filename is None:
            filename = f"{layer_name}.csv"
        df.to_csv(filename)


    def export_morpheus(self, filename='pdata', on='protein'):
        if not self._check_data(on):  # type: ignore[attr-defined], ValidationMixin
            return

        adata = self.prot if on == 'protein' else self.pep

        # alternatively, use morpheus to plot clustermap
        # will need two things
        # 1. dataset (proteins in column, samples in rows)
        dense_matrix = adata.X.toarray()
        df = pd.DataFrame(dense_matrix, index=adata.obs_names, columns=adata.var_names)
        df.to_csv(f'{filename}_protein_matrix.csv')
        # 2. File Annotations (each sample in a row, different annotations in columns)
        adata.obs.to_csv(f'{filename}_protein_annotations.csv')
        # 3. Protein Annotations (each protein in a row, different annotations in columns)
        adata.var.to_csv(f'{filename}_protein_annotations.csv')

        print(f"{format_log_prefix('result')} Morpheus export complete.")

    def _set_RS(self, rs, debug=False, validate=True):
        """
        Set the RS (protein × peptide) mapping matrix.

        This internal method assigns a new RS matrix to the object. If the input appears 
        to be in peptide × protein format, it will be automatically transposed.

        Args:
            rs (np.ndarray or sparse matrix): The new RS matrix to assign.
            debug (bool): If True, prints diagnostic information.
            validate (bool): If True (default), checks that the RS shape matches `.prot` and `.pep`.

        Returns:
            None
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
            print(f"{format_log_prefix('result',indent=1)} RS matrix set: {self._rs.shape} (proteins × peptides), sparsity: {sparsity:.2f}%")
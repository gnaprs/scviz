import numpy as np
import pandas as pd
import warnings
from scipy import sparse
from scviz import utils
from scviz.utils import format_log_prefix

class EditingMixin:
    """
    Handles direct editing and exporting of data matrices and metadata.

    Functions:
        set_X: Updates `.X` matrix for `.prot` or `.pep`.
        get_abundance: Retrieves abundance values from a specified layer.
        export: Exports AnnData matrices and metadata to files.
    """
    def set_X(self, layer, on = 'protein'):
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
                print(f"{format_log_prefix('info_only', indent=2)} Set {on} data to layer {layer}.")

            self._history.append(f"{on}: Set X to layer {layer}.") # type: ignore[attr-defined]

    def get_abundance(self, namelist=None, layer='X', on='protein',
                    classes=None, log=True, x_label='gene'):
        """
        Extract long-form abundance DataFrame from a pAnnData object.

        Parameters:
            pdata: pAnnData object
            namelist: list of accessions or genes to extract (optional)
            layer: which data layer to use (default: 'X')
            on: 'protein' or 'peptide'
            classes: obs column or list of columns to group by
            log: whether to apply log2 transform
            x_label: 'gene' or 'accession'

        Returns:
            pd.DataFrame with abundance + metadata
        """

        gene_to_acc, _ = self.get_gene_maps(on='protein' if on == 'peptide' else on) # type: ignore[attr-defined]


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
            print(f"{format_log_prefix('result',indent=1)} RS matrix set: {self._rs.shape} (proteins × peptides), sparsity: {sparsity:.2f}%")
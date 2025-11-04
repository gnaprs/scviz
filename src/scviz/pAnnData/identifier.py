from scviz.utils import format_log_prefix
import warnings
import pandas as pd
from scviz import utils

class IdentifierMixin:
    """
    Handles mapping between genes, accessions, and peptides.

    This mixin provides utilities for:
    
    - Building and caching bidirectional mappings:
        * For proteins: gene ↔ accession  
        * For peptides: peptide ↔ protein accession

    - Updating or refreshing identifier maps manually or via UniProt
    - Automatically filling in missing gene names using the UniProt API

    These mappings are cached and used throughout the `pAnnData` object to support resolution of user queries and consistent gene-accession-peptide tracking.

    Functions:
        _build_identifier_maps: Create forward/reverse maps based on protein or peptide data
        refresh_identifier_maps: Clear cached mappings to force rebuild
        get_identifier_maps: Retrieve (gene → acc, acc → gene) or (peptide ↔ protein) maps
        update_identifier_maps: Add or overwrite mappings (e.g., manual corrections)
        update_missing_genes: Fill missing gene names using the UniProt API
    """

    def _build_identifier_maps(self, adata, gene_col="Genes"):
        """
        Build bidirectional identifier mappings for genes/proteins or peptides/proteins.

        Depending on whether `adata` is `.prot` or `.pep`, this builds:

        - For proteins: gene ↔ accession
        - For peptides: peptide ↔ protein accession

        Args:
            adata (AnnData): Either `self.prot` or `self.pep`.
            gene_col (str): Column name in `.var` containing gene names (default: "Genes").

        Returns:
            tuple: A pair of dictionaries (`forward`, `reverse`) for identifier lookup.

        Note:
            For peptides, mapping relies on `utils.get_pep_prot_mapping()` to resolve protein accessions.

        Raises:
            Warning if peptide-to-protein mapping cannot be built.
        """
        from pandas import notna

        forward = {}
        reverse = {}

        if adata is self.prot:
            if gene_col in adata.var.columns:
                for acc, gene in zip(adata.var_names, adata.var[gene_col]):
                    if notna(gene):
                        gene = str(gene)
                        forward[gene] = acc
                        reverse[acc] = gene

        elif adata is self.pep:
            try:
                prot_acc_col = utils.get_pep_prot_mapping(self)
                pep_to_prot = adata.var[prot_acc_col]
                for pep, prot in zip(adata.var_names, pep_to_prot):
                    if notna(prot):
                        forward[prot] = pep
                        reverse[pep] = prot
            except Exception as e:
                warnings.warn(f"Could not build peptide-to-protein map: {e}")

        return forward, reverse

    def refresh_identifier_maps(self):
        """
        Clear cached identifier maps to force regeneration on next access.

        This removes the following attributes if present:
        
        - `_gene_maps_protein`: Gene ↔ Accession map for proteins
        - `_protein_maps_peptide`: Protein ↔ Peptide map for peptides

        Useful when `.var` annotations are updated and identifier mappings may have changed.
        """
        for attr in ["_gene_maps_protein", "_protein_maps_peptide"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def get_identifier_maps(self, on='protein'):
        """
        Retrieve gene/accession or peptide/protein mapping dictionaries.

        Depending on the `on` argument, returns a tuple of forward and reverse mappings:

        - If `on='protein'`: (gene → accession, accession → gene)

        - If `on='peptide'`: (protein accession → peptide, peptide → protein accession)

        Note: Alias `get_gene_maps()` also calls this function for compatibility.

        Args:
            on (str): Source of mapping. Must be `'protein'` or `'peptide'`.

        Returns:
            Tuple[dict, dict]: (forward mapping, reverse mapping)

        Raises:
            ValueError: If `on` is not `'protein'` or `'peptide'`.
        """
        if on in ('protein','prot'):
            return self._cached_identifier_maps_protein
        elif on in ('peptide','pep'):
            return self._cached_identifier_maps_peptide
        else:
            raise ValueError(f"Invalid value for 'on': {on}. Must be 'protein' or 'peptide'.")

    # TODO: add peptide remapping to var, but need to also update rs if you do this.
    def update_identifier_maps(self, mapping, on='protein', direction='forward', overwrite=False, verbose=True):
        """
        Update cached identifier maps with user-supplied mappings.

        This function updates the internal forward and reverse identifier maps
        for either proteins or peptides. Ensures consistency by updating both
        directions of the mapping.

        - For `'protein'`:
            * forward: gene → accession  
            * reverse: accession → gene

        - For `'peptide'`:
            * forward: protein accession → peptide
            * reverse: peptide → protein accession

        Args:
            mapping (dict): Dictionary of mappings to add.
            on (str): Which maps to update. Must be `'protein'` or `'peptide'`.
            direction (str): `'forward'` or `'reverse'` — determines how the `mapping` should be interpreted.
            overwrite (bool): If True, allows overwriting existing entries.
            verbose (bool): If True, prints a summary of updated keys.

        Note:
            The corresponding reverse map is automatically updated to maintain bidirectional consistency.

        Example:
            Add new gene-to-accession mappings (protein):
                ```python
                pdata.update_identifier_maps(
                    {'MYGENE1': 'P00001', 'MYGENE2': 'P00002'},
                    on='protein',
                    direction='forward'
                )
                ```

            Add peptide → protein mappings:
                ```python
                pdata.update_identifier_maps(
                    {'PEPTIDE_ABC': 'P12345'},
                    on='peptide',
                    direction='reverse'
                )
                ```

            Overwrite a protein → gene mapping:
                ```python
                pdata.update_identifier_maps(
                    {'P12345': 'NEWGENE'},
                    on='protein',
                    direction='reverse',
                    overwrite=True
                )
                ```

        """
        if on == 'protein':
            forward, reverse = self._cached_identifier_maps_protein
        elif on == 'peptide':
            forward, reverse = self._cached_identifier_maps_peptide
        else:
            raise ValueError(f"Invalid value for 'on': {on}. Must be 'protein' or 'peptide'.")

        source_map = forward if direction == 'forward' else reverse
        target_map = reverse if direction == 'forward' else forward

        added, updated, skipped = 0, 0, 0

        for key, val in mapping.items():
            if key in source_map:
                if overwrite:
                    source_map[key] = val
                    target_map[val] = key
                    updated += 1
                else:
                    skipped += 1
            else:
                source_map[key] = val
                target_map[val] = key
                added += 1

        message = (
            f"[update_identifier_maps] Updated '{on}' ({direction}): "
            f"{added} added, {updated} overwritten, {skipped} skipped."
        )

        if verbose:
            print(message)
        self._append_history(message)

        # Update .prot.var["Genes"] if updating protein identifier reverse map (accession → gene)
        if on == 'protein' and direction == 'reverse':
            updated_var_count = 0
            updated_accessions = []

            for acc, gene in mapping.items():
                if acc in self.prot.var_names:
                    self.prot.var.at[acc, "Genes"] = gene
                    updated_accessions.append(acc)
                    updated_var_count += 1

            if updated_var_count > 0:
                var_message = (
                    f"🔁 Updated `.prot.var['Genes']` for {updated_var_count} entries from custom mapping. "
                    f"(View details in `pdata.metadata['identifier_map_history']`)"
                )
                if verbose:
                    print(var_message)
                self._append_history(var_message)

        # Log detailed update history for all cases
        import datetime

        record = {
            'on': on,
            'direction': direction,
            'input_mapping': dict(mapping),  # shallow copy
            'overwrite': overwrite,
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'summary': {
                'added': added,
                'updated': updated,
                'skipped': skipped,
            }
        }

        if on == 'protein' and direction == 'reverse':
            record['updated_var_column'] = {
                'column': 'Genes',
                'accessions': updated_accessions,
                'n_updated': updated_var_count
            }

        self.metadata.setdefault("identifier_map_history", []).append(record)

    get_gene_maps = get_identifier_maps

    def update_missing_genes(self, gene_col="Genes", verbose=True):
        """
        Fill missing gene names in `.prot.var` using UniProt API.

        This function searches for missing values in the specified gene column
        and attempts to fill them by querying the UniProt API using protein
        accession IDs. If a gene name cannot be found, a placeholder
        'UNKNOWN_<accession>' is used instead.

        Args:
            gene_col (str): Column name in `.prot.var` to update (default: "Genes").
            verbose (bool): Whether to print summary information (default: True).

        Returns:
            None

        Note:
            - This function only operates on `.prot.var`, not `.pep.var`.
            - If UniProt is unavailable or returns no match, the missing entry is filled as `'UNKNOWN_<accession>'`.
            - To manually correct unknown entries later, use `update_identifier_maps()` with `direction='reverse'`.

        Example:
            Automatically fill missing gene names using UniProt:
                ```python
                pdata.update_missing_genes()
                ```
        """
        var = self.prot.var

        if gene_col not in var.columns:
            if verbose:
                print(f"{format_log_prefix('warn')} Column '{gene_col}' not found in .prot.var.")
            return

        missing_mask = var[gene_col].isna()
        if not missing_mask.any():
            if verbose:
                print(f"{format_log_prefix('result')} No missing gene names found.")
            return

        accessions = var.index[missing_mask].tolist()
        if verbose:
            print(f"{format_log_prefix('info_only')} {len(accessions)} proteins with missing gene names.")

        try:
            df = utils.get_uniprot_fields(
                accessions,
                search_fields=["accession", "gene_primary"],
                standardize=True
            )
        except Exception as e:
            print(f"{format_log_prefix('error')} UniProt query failed: {e}")
            return
        df = utils.standardize_uniprot_columns(df)

        if df.empty or "accession" not in df.columns or "gene_primary" not in df.columns:
            print(f"{format_log_prefix('warn')} UniProt returned no usable gene mapping columns.")
            return

        gene_map = dict(zip(df["accession"], df["gene_primary"]))
        filled = self.prot.var.loc[missing_mask].index.map(lambda acc: gene_map.get(acc))
        final_genes = [
            gene if pd.notna(gene) else f"UNKNOWN_{acc}"
            for acc, gene in zip(self.prot.var.loc[missing_mask].index, filled)
        ]
        self.prot.var.loc[missing_mask, gene_col] = final_genes

        found = sum(pd.notna(filled))
        unknown = len(final_genes) - found
        if verbose:
            if found:
                print(f"{format_log_prefix('result')} Recovered {found} gene name(s) from UniProt. Genes found:")
                filled_clean = [str(g) for g in filled if pd.notna(g)]
                preview = ", ".join(filled_clean[:10])
                if found > 10:
                    preview += "..."
                print("        ", preview)
            if unknown:
                missing_ids = self.prot.var.loc[missing_mask].index[pd.isna(filled)]
                print(f"{format_log_prefix('warn')} {unknown} gene name(s) still missing. Assigned as 'UNKNOWN_<accession>' for:")
                print("        ", ", ".join(missing_ids[:5]) + ("..." if unknown > 10 else ""))
                print("     💡 Tip: You can update these using `pdata.update_identifier_maps({'GENE': 'ACCESSION'}, on='protein', direction='reverse', overwrite=True)`\n")

    def search_annotations(self, query, on='protein', search_columns=None, case=False, return_all_matches=True):
        """
        Search protein or peptide annotations for matching biological terms.

        This function scans `.prot.var` or `.pep.var` for entries containing the provided keyword(s),
        across common annotation fields.

        Args:
            query (str or list of str): Term(s) to search for (e.g., "keratin", "KRT").
            on (str): Whether to search `"protein"` or `"peptide"` annotations (default: `"protein"`).
            search_columns (list of str, optional): Columns to search in. Defaults to common biological fields.
            case (bool): Case-sensitive search (default: False).
            return_all_matches (bool): If True, return matches from any column. If False, returns only rows that match all terms.

        Returns:
            pd.DataFrame: Filtered dataframe with a `Matched` column (True/False) and optionally match columns per term.

        Example:
            ```python
            pdata.search_annotations("keratin")
            pdata.search_annotations(["keratin", "cytoskeleton"], on="peptide", case=False)
            ```
        """
        import pandas as pd

        adata = self.prot if on == "protein" else self.pep
        df = adata.var.copy()

        if search_columns is None:
            search_columns = [
                "Accession", "Description", "Biological Process", "Cellular Component",
                "Molecular Function", "Genes", "Gene ID", "Reactome Pathways"
            ]

        # Ensure index is available as a searchable column
        df = df.copy()
        df["Accession"] = df.index.astype(str)

        # Convert query to list
        if isinstance(query, str):
            query = [query]

        # Search logic
        def match_func(val, term):
            if pd.isnull(val):
                return False
            return term in val if case else term.lower() in str(val).lower()

        match_results = pd.DataFrame(index=df.index)

        for term in query:
            per_col_match = pd.DataFrame({
                col: df[col].apply(match_func, args=(term,)) if col in df.columns else False
                for col in search_columns
            })
            row_match = per_col_match.any(axis=1)
            match_results[f"Matched_{term}"] = row_match

        if return_all_matches:
            matched_any = match_results.any(axis=1)
        else:
            matched_any = match_results.all(axis=1)

        result_df = df.copy()
        result_df["Matched"] = matched_any
        for col in match_results.columns:
            result_df[col] = match_results[col]

        return result_df[result_df["Matched"]]

from scviz.utils import format_log_prefix
import warnings
import pandas as pd
from scviz import utils

class IdentifierMixin:
    """
    Mixin for handling identifier mappings between genes, proteins, and peptides.
    Provides methods to build, refresh, and update identifier maps, as well as retrieve them.
    
    Functions:
        _build_identifier_maps: Constructs internal gene/accession maps from `.var`.
        refresh_identifier_maps: Recomputes identifier mappings.
        get_identifier_maps: Returns cached maps (accession → gene or vice versa).
        update_identifier_maps: Updates stored identifier mappings.
        update_missing_genes: Fills in missing gene names using external databases.
    """

    def _build_identifier_maps(self, adata, gene_col="Genes"):
        """
        Builds bidirectional mapping for:
        - protein: gene ↔ accession
        - peptide: peptide ↔ protein accession

        Returns: (forward, reverse)
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
        Refresh all gene/accession map caches.
        """
        for attr in ["_gene_maps_protein", "_protein_maps_peptide"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def get_identifier_maps(self, on='protein'):
        """
        Returns identifier mapping dictionaries:
        - on='protein': (gene → accession, accession → gene)
        - on='peptide': (protein accession → peptide, peptide → protein accession)

        Alias: get_gene_maps() for compatibility.
        """
        if on == 'protein':
            return self._cached_identifier_maps_protein
        elif on == 'peptide':
            return self._cached_identifier_maps_peptide
        else:
            raise ValueError(f"Invalid value for 'on': {on}. Must be 'protein' or 'peptide'.")

    # TODO: add peptide remapping to var, but need to also update rs if you do this.
    def update_identifier_maps(self, mapping, on='protein', direction='forward', overwrite=False, verbose=True):
        """
        Update cached identifier maps with user-supplied mappings.

        Parameters:
            mapping (dict): Dictionary of mappings to add.
            on (str): 'protein' or 'peptide' — which set of maps to update.
            direction (str): 'forward' or 'reverse'.
                - For 'protein':
                    forward: gene → accession
                    reverse: accession → gene
                - For 'peptide':
                    forward: protein accession → peptide
                    reverse: peptide → protein accession
            overwrite (bool): If True, overwrite existing entries.
            verbose (bool): If True, print a summary of the update.

        This updates both the forward and reverse maps to maintain consistency.

        Examples:
        ---------
        # Add new gene-to-accession mappings (protein)
        pdata.update_identifier_maps({'MYGENE1': 'P00001', 'MYGENE2': 'P00002'}, on='protein', direction='forward')

        # Add peptide → protein mappings
        pdata.update_identifier_maps({'PEPTIDE_ABC': 'P12345'}, on='peptide', direction='reverse')

        # Overwrite a protein → gene mapping
        pdata.update_identifier_maps({'P12345': 'NEWGENE'}, on='protein', direction='reverse', overwrite=True)
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
        Fills missing gene names in .prot.var[gene_col] using UniProt API.
        If UniProt returns no match, fills with 'UNKNOWN_<accession>'.
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
            print(f"{format_log_prefix('search')} {len(accessions)} proteins with missing gene names. Querying UniProt...")

        try:
            df = utils.get_uniprot_fields(
                accessions,
                search_fields=["accession", "gene_primary"]
            )
        except Exception as e:
            print(f"{format_log_prefix('error')} UniProt query failed: {e}")
            return

        gene_map = dict(zip(df["Entry"], df["Gene Names (primary)"]))
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
                print("💡 Tip: You can update these using `pdata.update_identifier_maps({'GENE': 'ACCESSION'}, on='protein', direction='reverse', overwrite=True)`")


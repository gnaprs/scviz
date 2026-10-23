import datetime
import requests
import pandas as pd
import numpy as np
import time
from io import StringIO
from IPython.display import SVG, display
import tempfile
import os

from scpviz.utils import format_log_prefix, get_uniprot_fields
import scpviz.utils as utils

class EnrichmentMixin:
    """
    Provides methods for STRING-based functional and protein–protein interaction (PPI) enrichment.

    This mixin includes utilities for:

    - Running functional enrichment on differentially expressed or user-supplied gene lists.
    - Performing STRING PPI enrichment to identify interaction networks.
    - Generating STRING network visualization links and embedded SVGs.
    - Listing and accessing enrichment results stored in `.stats`.

    Functions:
        enrichment_functional: Runs STRING functional enrichment on DE results or a custom gene list.
        enrichment_ppi: Runs STRING PPI enrichment on a user-supplied gene or accession list.
        list_enrichments: Lists available enrichment results and DE comparisons.
        plot_enrichment_svg: Displays a STRING enrichment SVG inline or saves it to file.
        get_string_mappings: Maps UniProt accessions to STRING IDs using the STRING API.
        update_string_mapping: Add or overwrite cached STRING IDs on `.prot.var`.
        resolve_to_accessions: Resolves gene names or mixed inputs to accessions using internal mappings.
        get_string_network_link: Generates a direct STRING network URL for visualization.
    """

    def get_string_mappings(self, identifiers, overwrite=False, cache_col="STRING", batch_size=100, debug=False):
        """
        Resolve STRING IDs for UniProt accessions with a 2-step strategy:
        1) Use UniProt stream (fields: xref_string) to fill cache quickly.
        2) For any still-missing rows, query STRING get_string_ids, batched by organism_id.

        This method retrieves corresponding STRING identifiers for a list of UniProt accessions
        and stores the result in `self.prot.var["STRING"]` for downstream use.

        Args:
            identifiers (list of str): List of UniProt accession IDs to map.
            overwrite (bool): If True, re-query even when a STRING ID is already cached.
            cache_col (str): `.prot.var` column used as the STRING ID cache (default: `"STRING"`).
            batch_size (int): Number of accessions to include in each API query (default is 100).
            debug (bool): If True, prints progress and response info.

        Returns:
            pd.DataFrame: Mapping table with columns: `input_identifier`, `string_identifier`, and `ncbi_taxon_id`.

        Note:
            This is a helper method used primarily by `enrichment_functional()` and `enrichment_ppi()`.
            Identifiers still unmapped after UniProt and STRING are printed so they can be looked
            up manually and added with `update_string_mapping()`.
        """

        identifiers = [str(x).strip() for x in identifiers if x is not None and str(x).strip()]
        if debug:
            print(f"{format_log_prefix('info')} Resolving STRING IDs for {len(identifiers)} identifiers...")

        prot_var = self.prot.var
        if cache_col not in prot_var.columns:
            prot_var[cache_col] = pd.NA
        if "ncbi_taxon_id" not in prot_var.columns:
            prot_var["ncbi_taxon_id"] = pd.NA
        
        # Use cached STRING IDs if available
        valid_ids = [i for i in identifiers if i in prot_var.index]
        existing = prot_var.loc[valid_ids, cache_col]
        found_ids = {i: sid for i, sid in existing.items() if pd.notna(sid) and str(sid).strip()}
        missing = [i for i in identifiers if i not in found_ids]

        if overwrite:
            print(f"{format_log_prefix('info_only',2)} Overwriting cached STRING IDs.")
            missing = valid_ids
            found_ids = {}

        print(f"{format_log_prefix('info_only',2)} Found {len(found_ids)} cached STRING IDs. {len(missing)} need lookup.")
        print(missing) if debug else None

        # 1. UniProt stream (fast)         # Use UniProt xref_string field to fill cache quickly
        # 2. STRING API for still-missing ones

        if missing:
            map_df = utils.get_string_mappings(
                missing,
                use_uniprot=True,
                use_string=True,
                caller_identity="scpviz",
                batch_size=batch_size,
                debug=debug,
            )
        else:
            map_df = pd.DataFrame(columns=["input_identifier", "string_identifier", "ncbi_taxon_id"])

        # Combine all new mappings
        if not map_df.empty:
            updated = 0
            updated_tax = 0

            for _, row in map_df.iterrows():
                acc = row["input_identifier"]
                sid = row["string_identifier"]
                tax = row.get("ncbi_taxon_id", pd.NA)
                
                if acc is None or acc not in prot_var.index:
                    continue

                if pd.notna(sid) and str(sid).strip():
                    self.prot.var.at[acc, cache_col] = sid
                    found_ids[acc] = sid
                    updated += 1
                else:
                    print(f"[DEBUG] Skipping unknown accession '{acc}'") if debug else None

                tax = utils.scalarize_taxon(tax)
                if tax is not pd.NA and pd.notna(tax):
                    prot_var.at[acc, "ncbi_taxon_id"] = str(tax)
                    updated_tax += 1

            print(f"{format_log_prefix('info_only',3)} Cached {updated} STRING ID mappings.")
            if debug:
                print(f"{format_log_prefix('info_only',3)} Cached {updated_tax} ncbi_taxon_id values.")

        elif missing:
            print(f"{format_log_prefix('warn_only',3)} No STRING mappings returned from STRING API.")

        leftover = []
        for acc in identifiers:
            sid = found_ids.get(
                acc,
                prot_var.at[acc, cache_col] if acc in prot_var.index else pd.NA,
            )
            if not (pd.notna(sid) and str(sid).strip()):
                leftover.append(acc)
        if leftover:
            preview_n = 20
            preview = ", ".join(leftover[:preview_n])
            extra = f" ... (+{len(leftover) - preview_n} more)" if len(leftover) > preview_n else ""
            print(
                f"{format_log_prefix('warn_only', 3)} {len(leftover)} identifier(s) still unmapped: {preview}{extra}"
            )
            print(
                f"{format_log_prefix('blank', 3)}Tip: look these up on STRING, then "
                f"`pdata.update_string_mapping({{'ACCESSION': '9606.ENSP...'}})`."
            )

        # ------------------------------------
        # Build and MERGE UniProt results into out_df
        # ------------------------------------
        out_df = pd.DataFrame({"input_identifier": identifiers})

        # Prefer found_ids (fresh + cached), else fall back to cache for known indices
        out_df["string_identifier"] = out_df["input_identifier"].map(
            lambda acc: found_ids.get(
                acc,
                prot_var.at[acc, cache_col] if acc in prot_var.index else pd.NA,
            )
        )

        out_df["ncbi_taxon_id"] = out_df["input_identifier"].map(
            lambda acc: prot_var.at[acc, "ncbi_taxon_id"] if acc in prot_var.index else pd.NA
        )
        out_df["ncbi_taxon_id"] = out_df["ncbi_taxon_id"].apply(utils.scalarize_taxon)

        return out_df


    def update_string_mapping(
        self, mapping, overwrite=False, verbose=True, taxon=None, cache_col="STRING"
    ):
        """
        Update cached STRING IDs on `.prot.var` with user-supplied mappings.

        Use this after `get_string_mappings()` reports identifiers that UniProt and STRING
        could not resolve. New or empty cache entries are filled even when `overwrite=False`;
        existing STRING IDs are left unchanged unless `overwrite=True`.

        Args:
            mapping (dict): Dictionary of `{accession: STRING_id}` mappings to add.
            overwrite (bool): If True, replace existing cached STRING IDs. Default False.
            verbose (bool): If True, print a summary of added, overwritten, and skipped keys.
            taxon (int, str, or dict, optional): NCBI taxonomy ID to write to
                `.prot.var["ncbi_taxon_id"]` for keys that are added or overwritten.
                A scalar applies to all updated accessions; a dict maps accession → taxon.
            cache_col (str): `.prot.var` column used as the STRING ID cache (default: `"STRING"`).

        Returns:
            None

        Raises:
            ValueError: If `mapping` is not a dictionary.

        Example:
            Add STRING IDs for accessions that were not mapped automatically:
                ```python
                pdata.update_string_mapping({
                    "P12345": "9606.ENSP00000229277",
                    "Q9Y2X3": "9606.ENSP00000312345",
                })
                ```

            Overwrite an existing cached STRING ID and set taxon:
                ```python
                pdata.update_string_mapping(
                    {"P12345": "9606.ENSP00000229277"},
                    taxon=9606,
                    overwrite=True,
                )
                ```
        """
        if not isinstance(mapping, dict):
            raise ValueError("`mapping` must be a dict of {accession: STRING_id}.")

        prot_var = self.prot.var
        if cache_col not in prot_var.columns:
            prot_var[cache_col] = pd.NA
        if "ncbi_taxon_id" not in prot_var.columns:
            prot_var["ncbi_taxon_id"] = pd.NA

        added, updated, skipped = 0, 0, 0
        unknown = []

        for acc, sid in mapping.items():
            acc = str(acc).strip()
            sid_ok = pd.notna(sid) and str(sid).strip()
            if acc not in prot_var.index:
                unknown.append(acc)
                skipped += 1
                continue
            if not sid_ok:
                skipped += 1
                continue

            existing = prot_var.at[acc, cache_col]
            has_existing = pd.notna(existing) and str(existing).strip()
            if has_existing and not overwrite:
                skipped += 1
                continue

            prot_var.at[acc, cache_col] = str(sid).strip()
            if has_existing:
                updated += 1
            else:
                added += 1

            tax_val = taxon.get(acc) if isinstance(taxon, dict) else taxon
            if tax_val is not None:
                tax_val = utils.scalarize_taxon(tax_val)
                if tax_val is not pd.NA and pd.notna(tax_val):
                    prot_var.at[acc, "ncbi_taxon_id"] = str(tax_val)

        message = (
            f"prot: STRING mapping update: {added} added, {updated} overwritten, {skipped} skipped."
        )
        if verbose:
            print(f"{format_log_prefix('update')} STRING mapping: {added} added, {updated} overwritten, {skipped} skipped.")
            if unknown:
                preview_n = 20
                preview = ", ".join(unknown[:preview_n])
                extra = f" ... (+{len(unknown) - preview_n} more)" if len(unknown) > preview_n else ""
                print(
                    f"{format_log_prefix('warn')} Accessions not in .prot.var (skipped): {preview}{extra}"
                )
        self._append_history(message)  # type: ignore[attr-defined], HistoryMixin

        record = {
            "cache_col": cache_col,
            "input_mapping": dict(mapping),
            "overwrite": overwrite,
            "taxon": taxon if not isinstance(taxon, dict) else dict(taxon),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "summary": {
                "added": added,
                "updated": updated,
                "skipped": skipped,
                "unknown": list(unknown),
            },
        }
        self.metadata.setdefault("string_map_history", []).append(record)

    def resolve_to_accessions(self, mixed_list):
        """
        Convert gene names or accessions into standardized UniProt accession IDs.

        This method resolves input items using the internal gene-to-accession map,
        ensuring all returned entries are accessions present in the `.prot` object.

        Args:
            mixed_list (list of str): A list containing gene names and/or UniProt accessions.

        Returns:
            list of str: List of resolved UniProt accession IDs.

        Note:
            This function is similar to `utils.resolve_accessions()` but operates in the context 
            of the current `pAnnData` object and its internal gene mappings.

        Todo:
            Add example comparing results from `resolve_to_accessions()` and `utils.resolve_accessions()`.
        """
        gene_to_acc, _ = self.get_gene_maps(on='protein') 
        accs = []
        unresolved_accs = []
        for item in mixed_list:
            if item in self.prot.var.index:
                accs.append(item)  # already an accession
            elif item in gene_to_acc:
                accs.append(gene_to_acc[item])
            else:
                unresolved_accs.append(item)
                # print(f"{format_log_prefix('warn_only',2)} Could not resolve '{item}' to an accession — skipping.")
        return accs, unresolved_accs

    def enrichment_functional(
        self,
        genes=None,
        from_de=True,
        top_n=None,
        score_col="significance_score",
        gene_col="Genes",
        de_key="de_results",
        store_key=None,
        species=None,
        background=None,
        debug=False,
        **kwargs
    ):
        """
        Run functional enrichment analysis using STRING on a gene list.

        This method performs ranked or unranked enrichment analysis using STRING's API.
        It supports both differential expression-based analysis (up- and down-regulated genes)
        and custom gene lists provided by the user. Enrichment results are stored in
        `.stats["functional"]` for later access and plotting.

        Args:
            genes (list of str, optional): List of gene symbols to analyze. Ignored if `from_de=True`.
            from_de (bool): If True (default), selects genes from stored differential expression results.
            top_n (int, optional): If set, keep this many top-ranked genes per tail when
                `from_de=True`, ranked by `score_col`. Default `None` uses all significant
                DE genes (upregulated and downregulated).
            score_col (str): Column name in the DE table to rank genes by (default is `"significance_score"`).
            gene_col (str): Column name in `.prot.var` or DE results that contains gene names.
            de_key (str): Key to retrieve stored DE results from `.stats["de_results"]`.
            store_key (str, optional): Custom key to store enrichment results. Ignored when `from_de=True`.
            species (str, optional): Organism name or NCBI taxonomy ID. If None, inferred from STRING response.
            background (str or list of str, optional): Background gene list to use for enrichment.

                - If `"all_quantified"`, uses non-significant proteins from DE or all other quantified proteins.
                - If a list, must contain valid gene names or accessions.
            debug (bool): If True, prints API request info and diagnostic messages.
            **kwargs: Additional keyword arguments passed to the STRING enrichment API.

        Returns:
            dict or pd.DataFrame:

                - If `from_de=True`, returns a dictionary of enrichment DataFrames for "up" and "down" gene sets.
                - If `genes` is provided, returns a single enrichment DataFrame.

        Example:
            Run differential expression, then perform STRING enrichment on all significant genes:
                ```python
                case1 = {'cellline': 'AS', 'treatment': 'sc'} # legacy style: class_type = ["group", "condition"]
                case2 = {'cellline': 'BE', 'treatment': 'sc'} # legacy style: values = [["GroupA", "Treatment1"], ["GroupA", "Control"]]
                pdata_nb.de(values = case_values) # or legacy style: pdata.de(classes=class_type, values=values)
                pdata.list_enrichments()  # list available DE result keys
                pdata.enrichment_functional(from_de=True, de_key="GroupA_Treatment1 vs GroupA_Control")
                ```

            Perform enrichment on a custom list of genes:
                ```python
                genelist = ["P55072", "NPLOC4", "UFD1", "STX5A", "NSFL1C", "UBXN2A",
                            "UBXN4", "UBE4B", "YOD1", "WASHC5", "PLAA", "UBXN10"]
                pdata.enrichment_functional(genes=genelist, from_de=False)
                ```

        Note:
            Internally uses `resolve_to_accessions()` and `get_string_mappings()`, and stores results 
            in `.stats["functional"]`. Results can be accessed or visualized via `plot_enrichment_svg()`
            or by visiting the linked STRING URLs.
        """
        def query_functional_enrichment(query_ids, species_id, background_ids=None, debug=False):
            print(f"{format_log_prefix('info_only',2)} Running enrichment on {len(query_ids)} STRING IDs (species {species_id})...") if debug else None
            url = "https://string-db.org/api/json/enrichment"
            payload = {
                "identifiers": "%0d".join(query_ids),
                "species": species_id,
                "caller_identity": "scpviz"
            }
            if background_ids is not None:
                print(f"{format_log_prefix('info_only')} Using background of {len(background_ids)} STRING IDs.")
                payload["background_string_identifiers"] = "%0d".join(background_ids)

            print(payload) if debug else None
            response = requests.post(url, data=payload)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        
        # Ensure string metadata section exists
        if "functional" not in self.stats:
            self.stats["functional"] = {}

        if genes is None and from_de:
            resolved_key = _resolve_de_key(self.stats, de_key)
            de_df = self.stats[resolved_key]
            print(f"{format_log_prefix('user')} Running STRING enrichment [DE-based: {resolved_key}]")

            sig_up = de_df[de_df["significance"] == "upregulated"].copy()
            sig_down = de_df[de_df["significance"] == "downregulated"].copy()
            sig_up = sig_up[sig_up[gene_col].notna()]
            sig_down = sig_down[sig_down[gene_col].notna()]

            if top_n is not None:
                try:
                    n_keep = int(top_n)
                except (TypeError, ValueError) as exc:
                    raise ValueError("`top_n` must be a positive integer or None.") from exc
                if n_keep <= 0:
                    raise ValueError("`top_n` must be a positive integer or None.")
                print(
                    f"{format_log_prefix('info_only', 2)} Truncating to top {n_keep} genes per tail "
                    f"(by {score_col}; {len(sig_up)} up, {len(sig_down)} down before truncation)."
                )
                sig_up = sig_up.sort_values(score_col, ascending=False).head(n_keep)
                sig_down = sig_down.sort_values(score_col, ascending=True).head(n_keep)

            up_genes = sig_up[gene_col].tolist()
            down_genes = sig_down[gene_col].tolist()

            up_accs, up_unresolved = self.resolve_to_accessions(up_genes)
            down_accs, down_unresolved = self.resolve_to_accessions(down_genes)

            background_accs = None
            background_string_ids = None
            if background == "all_quantified":
                print(f"{format_log_prefix('warn')} Mapping background proteins may take a long time due to batching.")
                background_accs = de_df[de_df["significance"] == "not significant"].index.tolist()

            if background_accs:
                bg_map = self.get_string_mappings(background_accs,debug=debug)
                bg_map = bg_map[bg_map["string_identifier"].notna()]
                background_string_ids = bg_map["string_identifier"].tolist()

            if store_key is not None:
                print(f"{format_log_prefix('warn')} Ignoring `store_key` for DE-based enrichment. Using auto-generated pretty keys.")

            results = {}
            for label, accs in zip(["up", "down"], [up_accs, down_accs]):
                print(f"\n🔹 {label.capitalize()}-regulated proteins")
                t0 = time.time()

                if not accs:
                    print(f"{format_log_prefix('warn')} No {label}-regulated proteins to analyze.")
                    continue

                mapping_df = self.get_string_mappings(accs, debug=debug)
                mapping_df = mapping_df[mapping_df["string_identifier"].notna()]
                if mapping_df.empty:
                    print(f"{format_log_prefix('warn')} No valid STRING mappings found for {label}-regulated proteins.")
                    continue

                string_ids = mapping_df["string_identifier"].tolist()
                inferred_species = mapping_df["ncbi_taxon_id"].mode().iloc[0]
                if species is not None:
                    # check if user species is same as inferred
                    if inferred_species != species:
                        print(f"{format_log_prefix('warn',2)} Inferred species ({inferred_species}) does not match user-specified ({species}). Using user-specified species.")
                    species_id = species
                else:
                    species_id = inferred_species

                print(f"   🔸 Proteins: {len(accs)} → STRING IDs: {len(string_ids)}")
                print(f"   🔸 Species: {species_id} | Background: {'None' if background_string_ids is None else 'custom'}")
                if label == "up":
                    if up_unresolved:
                        print(f"{format_log_prefix('warn',2)} Some accessions unresolved for {label}-regulated proteins: {', '.join(up_unresolved)}")
                else:
                    if down_unresolved:
                        print(f"{format_log_prefix('warn',2)} Some accessions unresolved for {label}-regulated proteins: {', '.join(down_unresolved)}")

                enrichment_df = query_functional_enrichment(string_ids, species_id, background_string_ids, debug=debug)
                enrich_key = f"{resolved_key}_{label}"
                pretty_base = _pretty_vs_key(resolved_key)
                pretty_key = f"{pretty_base}_{label}"
                string_url = self.get_string_network_link(string_ids=string_ids, species=species_id)

                self.stats["functional"][pretty_key] = {
                    "string_ids": string_ids,
                    "background_string_ids": background_string_ids,
                    "species": species_id,
                    "input_key": resolved_key if from_de else None,
                    "string_url": string_url,
                    "result": enrichment_df
                }

                print(f"{format_log_prefix('result')} Enrichment complete ({time.time() - t0:.2f}s)")
                print(f"   • Access result: pdata.stats['functional'][\"{pretty_key}\"][\"result\"]")
                print(f"   • Plot command : pdata.plot_enrichment_svg(\"{pretty_base}\", direction=\"{label}\")")
                print(f"   • View online  : {string_url}\n")

                results[label] = enrichment_df

        elif genes is not None:
            t0 = time.time()
            print(f"{format_log_prefix('user')} Running STRING enrichment [user-supplied]")

            if store_key is None:
                prefix = "UserSearch"
                existing = self.stats["functional"].keys() if "functional" in self.stats else []
                existing_ids = [k for k in existing if k.startswith(prefix)]
                next_id = len(existing_ids) + 1
                store_key = f"{prefix}{next_id}"

            input_accs, unresolved_accs = self.resolve_to_accessions(genes)
            mapping_df = self.get_string_mappings(input_accs, debug=debug)
            mapping_df = mapping_df[mapping_df["string_identifier"].notna()]
            if mapping_df.empty:
                raise ValueError("No valid STRING mappings found for the provided identifiers.")

            string_ids = mapping_df["string_identifier"].tolist()
            inferred_species = mapping_df["ncbi_taxon_id"].mode().iloc[0]
            if species is not None:
                # check if user species is same as inferred
                if inferred_species != species:
                    print(f"{format_log_prefix('warn',2)} Inferred species ({inferred_species}) does not match user-specified ({species}). Using user-specified species.")
                species_id = species
            else:
                species_id = inferred_species

            background_string_ids = None
            if background == "all_quantified":
                print(f"{format_log_prefix('warn')} Mapping background proteins may take a long time due to batching.")
                all_accs = list(self.prot.var_names)
                background_accs = list(set(all_accs) - set(input_accs))
                bg_map = self.get_string_mappings(background_accs, debug=debug)
                bg_map = bg_map[bg_map["string_identifier"].notna()]
                background_string_ids = bg_map["string_identifier"].tolist()

            print(f"   🔸 Input genes: {len(genes)} → Resolved STRING IDs: {len(string_ids)}")
            print(f"   🔸 Species: {species_id} | Background: {'None' if background_string_ids is None else 'custom'}")
            if unresolved_accs:
                print(f"{format_log_prefix('warn',2)} Some accessions unresolved: {', '.join(unresolved_accs)}")

            enrichment_df = query_functional_enrichment(string_ids, species_id, background_string_ids, debug=debug)
            string_url = self.get_string_network_link(string_ids=string_ids, species=species_id)

            self.stats["functional"][store_key] = {
                "string_ids": string_ids,
                "background_string_ids": background_string_ids,
                "species": species_id,
                "input_key": None,
                "string_url": string_url,
                "result": enrichment_df
            }

            print(f"{format_log_prefix('result')} Enrichment complete ({time.time() - t0:.2f}s)")
            print(f"   • Access result: pdata.stats['functional'][\"{store_key}\"][\"result\"]")
            print(f"   • Plot command : pdata.plot_enrichment_svg(\"{store_key}\")")
            print(f"   • View online  : {string_url}\n")

            return enrichment_df

        else:
            raise ValueError("Must provide 'genes' or set from_de=True to use DE results.") 

    def enrichment_ppi(self, genes, species=None, store_key=None, debug=False):
        """
        Run STRING PPI (protein–protein interaction) enrichment on a user-supplied gene or accession list.

        This method maps the input gene names or UniProt accessions to STRING IDs, infers the species 
        if not provided, and submits the list to STRING's PPI enrichment endpoint. Results are stored 
        in `.stats["ppi"]` for later retrieval or visualization.

        Args:
            genes (list of str): A list of gene names or UniProt accessions to analyze.
            species (int or str, optional): NCBI taxonomy ID (e.g., 9606 for human). If None, inferred from STRING mappings.
            store_key (str, optional): Key to store the enrichment result under `.stats["ppi"]`.
                If None, a unique key is auto-generated.

        Returns:
            pd.DataFrame: DataFrame of STRING PPI enrichment results.

        Example:
            Run differential expression, then perform STRING PPI enrichment on significant genes:
                ```python
                class_type = ["group", "condition"]
                values = [["GroupA", "Treatment1"], ["GroupA", "Control"]]

                pdata.de(classes=class_type, values=values)
                pdata.list_enrichments()
                sig_genes = pdata.stats["de_results"]["GroupA_Treatment1 vs GroupA_Control"]
                sig_genes = sig_genes[sig_genes["significance"] != "not significant"]["Genes"].dropna().tolist()

                pdata.enrichment_ppi(genes=sig_genes)
                ```
        """
        def query_ppi_enrichment(string_ids, species):
            # print(f"[INFO] Running PPI enrichment for {len(string_ids)} STRING IDs (species {species})...")
            url = "https://string-db.org/api/json/ppi_enrichment"
            payload = {
                "identifiers": "%0d".join(string_ids),
                "species": species,
                "caller_identity": "scpviz"
            }

            response = requests.post(url, data=payload)
            response.raise_for_status()

            result = response.json()
            print("[DEBUG] PPI enrichment result:", result)
            return result[0] if isinstance(result, list) else result

        print(f"{format_log_prefix('user')} Running STRING PPI enrichment")
        t0 = time.time()
        input_accs, unresolved_accs = self.resolve_to_accessions(genes)
        mapping_df = self.get_string_mappings(input_accs, debug=debug)
        mapping_df = mapping_df[mapping_df["string_identifier"].notna()]
        if mapping_df.empty:
            raise ValueError("No valid STRING mappings found for the provided genes/accessions.")

        string_ids = mapping_df["string_identifier"].tolist()
        inferred_species = mapping_df["ncbi_taxon_id"].mode().iloc[0]
        species_id = species if species is not None else inferred_species

        print(f"   🔸 Input genes: {len(genes)} → Resolved STRING IDs: {len(mapping_df)}")
        print(f"   🔸 Species: {species_id}")
        if unresolved_accs:
            print(f"{format_log_prefix('warn', 2)} Some accessions unresolved: {', '.join(unresolved_accs)}")

        result = query_ppi_enrichment(string_ids, species_id)

        # Store results
        if "ppi" not in self.stats:
            self.stats["ppi"] = {}

        if store_key is None:
            base = "UserPPI"
            counter = 1
            while f"{base}{counter}" in self.stats["ppi"]:
                counter += 1
            store_key = f"{base}{counter}"

        self.stats["ppi"][store_key] = {
            "result": result,
            "string_ids": string_ids,
            "species": species_id
        }

        print(f"{format_log_prefix('result')} PPI enrichment complete ({time.time() - t0:.2f}s)")
        print(f"   • STRING IDs   : {len(string_ids)}")
        print(f"   • Edges found  : {result['number_of_edges']} vs {result['expected_number_of_edges']} expected")
        print(f"   • p-value      : {result['p_value']:.2e}")
        print(f"   • Access result: pdata.stats['ppi']['{store_key}']['result']\n")

        return result

    def enrichment_functional_pca(
        self,
        pcs=(1, 2),
        on="protein",
        top_n=150,
        gene_col="Genes",
        direction="both",
        key_prefix="PCA",
        species=None,
        background=None,
        debug=False,
        **kwargs,
    ):
        """
        Run STRING functional enrichment on top PCA-loading proteins/genes.

        This method selects top features from PCA loadings for each requested PC and
        delegates STRING querying/storage to ``enrichment_functional(from_de=False)``.

        Args:
            pcs (iterable of int): PCs to analyze (1-indexed).
            on (str): Data level, either ``"protein"`` or ``"peptide"``.
            top_n (int): Number of features to keep. For ``direction="both"``, the top
                ``top_n`` genes by absolute loading (one enrichment per PC). For
                ``"positive"`` or ``"negative"``, the top ``top_n`` in that tail only.
            gene_col (str): Column in ``.var`` used for pathway labels/genes.
                Falls back to ``.var_names`` when missing.
            direction (str): ``"both"`` runs one STRING enrichment per PC on the top
                ``top_n`` genes ranked by ``|loading|``. ``"positive"`` / ``"negative"``
                run separate enrichments on the top ``top_n`` positive or negative
                loadings only.
            key_prefix (str): Prefix used for stored keys in ``.stats["functional"]``.
            species (str or int, optional): Passed to STRING enrichment.
            background (str or list, optional): Passed to STRING enrichment.
            debug (bool): Verbose diagnostics.
            **kwargs: Additional kwargs forwarded to ``enrichment_functional``.

        Returns:
            dict: Nested dictionary of results, keyed by ``"PCk"`` then either ``{"abs": df}``
            (when ``direction="both"``) or ``{"positive": df}`` / ``{"negative": df}``.
        """
        if on == "protein":
            adata = self.prot
        elif on == "peptide":
            adata = self.pep
        else:
            raise ValueError("`on` must be either 'protein' or 'peptide'.")

        if "pca" not in adata.uns or "PCs" not in adata.uns["pca"]:
            raise ValueError(f"PCA results not found in .{ 'prot' if on == 'protein' else 'pep' }.uns['pca'].")

        if direction not in {"both", "positive", "negative"}:
            raise ValueError("`direction` must be one of {'both', 'positive', 'negative'}.")

        pcs_all = adata.uns["pca"]["PCs"]
        n_pcs_available = pcs_all.shape[0]
        pcs = [int(pc) for pc in pcs]
        invalid = [pc for pc in pcs if pc < 1 or pc > n_pcs_available]
        if invalid:
            raise ValueError(
                f"Invalid PCs requested: {invalid}. Available PCs are 1 to {n_pcs_available}."
            )
        if int(top_n) <= 0:
            raise ValueError("`top_n` must be a positive integer.")

        genes = (
            adata.var[gene_col].copy()
            if gene_col in adata.var.columns
            else pd.Series(adata.var_names.astype(str), index=adata.var_names)
        )
        if gene_col not in adata.var.columns:
            print(f"{format_log_prefix('warn')} `.var[{gene_col!r}]` not found. Falling back to `.var_names`.")

        def _next_available_key(base):
            functional = self.stats.get("functional", {})
            if base not in functional:
                return base
            i = 2
            while f"{base}_{i}" in functional:
                i += 1
            return f"{base}_{i}"

        print(f"{format_log_prefix('user')} Running STRING enrichment [PCA-loading based]")
        print(f"   🔸 PCs: {pcs} | top_n: {top_n} | direction: {direction}")

        all_results = {}
        for pc in pcs:
            loadings = pcs_all[pc - 1, :]
            rank_df = pd.DataFrame(
                {"feature": adata.var_names.astype(str), "gene": genes.values, "loading": loadings}
            ).dropna(subset=["gene"]).copy()
            rank_df["gene"] = rank_df["gene"].astype(str)
            # Collapse duplicate gene labels by strongest absolute loading.
            idx = rank_df.groupby("gene")["loading"].agg(lambda x: x.abs().idxmax())
            rank_df = rank_df.loc[idx].copy()

            pc_results = {}
            if direction == "both":
                rank_df = rank_df.assign(abs_loading=rank_df["loading"].abs())
                abs_genes = (
                    rank_df.sort_values("abs_loading", ascending=False)["gene"]
                    .head(int(top_n))
                    .tolist()
                )
                key = _next_available_key(f"{key_prefix}_PC{pc}_abs")
                pc_results["abs"] = self.enrichment_functional(
                    genes=abs_genes,
                    from_de=False,
                    store_key=key,
                    species=species,
                    background=background,
                    debug=debug,
                    **kwargs,
                )
            else:
                if direction == "positive":
                    pos_genes = rank_df.sort_values("loading", ascending=False)["gene"].head(int(top_n)).tolist()
                    key = _next_available_key(f"{key_prefix}_PC{pc}_positive")
                    pc_results["positive"] = self.enrichment_functional(
                        genes=pos_genes,
                        from_de=False,
                        store_key=key,
                        species=species,
                        background=background,
                        debug=debug,
                        **kwargs,
                    )
                else:
                    neg_genes = rank_df.sort_values("loading", ascending=True)["gene"].head(int(top_n)).tolist()
                    key = _next_available_key(f"{key_prefix}_PC{pc}_negative")
                    pc_results["negative"] = self.enrichment_functional(
                        genes=neg_genes,
                        from_de=False,
                        store_key=key,
                        species=species,
                        background=background,
                        debug=debug,
                        **kwargs,
                    )

            all_results[f"PC{pc}"] = pc_results

        return all_results

    def list_enrichments(self):
        """
        List available STRING enrichment results and unprocessed DE contrasts.

        This method prints available functional and PPI enrichment entries stored in
        `.stats["functional"]` and `.stats["ppi"]`, as well as DE comparisons in 
        `.stats["de_results"]` that have not yet been analyzed.

        Returns:
            None

        Example:
            List enrichment results stored after running functional or PPI enrichment:
                ```python
                pdata.list_enrichments()
                ```
        """

        functional = self.stats.get("functional", {})
        ppi_keys = self.stats.get("ppi", {}).keys()
        de_keys = {k for k in self.stats if "vs" in k and not k.endswith(("_up", "_down"))}

        # Collect enriched DE keys based on input_key metadata
        enriched_de = set()
        enriched_results = []

        for k, meta in functional.items():
            input_key = meta.get("input_key", None)
            is_de = "vs" in k

            if input_key and input_key in de_keys:
                base = input_key
                suffix = k.rsplit("_", 1)[-1]
                pretty = f"{_pretty_vs_key(base)}_{suffix}"
                enriched_de.add(base)
                enriched_results.append((pretty, k, "DE-based"))
            else:
                enriched_results.append((k, k, "User"))

        de_unenriched = sorted(_pretty_vs_key(k) for k in (de_keys - enriched_de))

        print(f"{format_log_prefix('user')} Listing STRING enrichment status\n")

        print(f"{format_log_prefix('info_only',2)} Available DE comparisons (not yet enriched):")
        if de_unenriched:
            for pk in de_unenriched:
                print(f"        - {pk}")
        else:
            print("  (none)\n")

        print("\n  🔹 To run enrichment:")
        print("      pdata.enrichment_functional(from_de=True, de_key=\"...\")")
        
        print(f"\n{format_log_prefix('result_only')} Completed STRING enrichment results:")
        if not enriched_results:
            print("    (none)")
        for pretty, raw_key, kind in enriched_results:
            if kind == "DE-based":
                base, suffix = pretty.rsplit("_", 1)
                print(f"  - {pretty} ({kind})")
                print(f"    • Table: pdata.stats['functional'][\"{raw_key}\"]['result']")
                print(f"    • Plot : pdata.plot_enrichment_svg(\"{base}\", direction=\"{suffix}\")")
                url = self.stats["functional"].get(raw_key, {}).get("string_url")
                if url:
                    print(f"    • Link  : {url}")
            else:
                print(f"  - {pretty} ({kind})")
                print(f"    • Table: pdata.stats['functional'][\"{raw_key}\"]['result']")
                print(f"    • Plot : pdata.plot_enrichment_svg(\"{pretty}\")")
                url = self.stats["functional"].get(raw_key, {}).get("string_url")
                if url:
                    print(f"    • Link  : {url}")

        if ppi_keys:
            print(f"\n{format_log_prefix('result_only')} Completed STRING enrichment results:")
            for key in sorted(ppi_keys):
                print(f"  - {key} (User)")
                print(f"    • Table: pdata.stats['ppi']['{key}']['result']")
        else:
            print(f"\n{format_log_prefix('result_only')} Completed STRING PPI results:")
            print("    (none)")

    def plot_enrichment_svg(self, key, direction=None, category=None, save_as=None):
        """
        Display STRING enrichment SVG inline in a Jupyter notebook.

        This method fetches and renders a STRING-generated SVG for a previously completed
        functional enrichment result. Optionally, the SVG can also be saved to disk.

        Args:
            key (str): Enrichment result key from `.stats["functional"]`. For DE-based comparisons, this 
                includes both contrast and direction (e.g., `"GroupA_Treatment1_vs_Control_up"`).
            direction (str, optional): Direction of DE result, either `"up"` or `"down"`. Use `None` for 
                user-defined gene lists.
            category (str, optional): STRING enrichment category to filter by (e.g., `"Process"`, `"KEGG"`). See the table in the below note for options.
            save_as (str, optional): If provided, saves the retrieved SVG to the given file path.

        Returns:
            None

        Example:
            Display a STRING enrichment network for a user-supplied gene list:
                ```python
                pdata.plot_enrichment_svg("UserSearch1")
                ```

        !!! note "Supported STRING Enrichment Categories"
            The following category IDs are supported for functional enrichment.  
            More details are available on the [STRING API documentation site](https://string-db.org/cgi/help?subpage=api).

            | Category ID          | Description                                      |
            |----------------------|--------------------------------------------------|
            | **Process**          | Biological Process (Gene Ontology)               |
            | **Function**         | Molecular Function (Gene Ontology)               |
            | **Component**        | Cellular Component (Gene Ontology)               |
            | **Keyword**          | Annotated Keywords (UniProt)                     |
            | **KEGG**             | KEGG Pathways                                    |
            | **RCTM**             | Reactome Pathways                                |
            | **HPO**              | Human Phenotype (Monarch)                        |
            | **MPO**              | Mammalian Phenotype Ontology (Monarch)           |
            | **DPO**              | Drosophila Phenotype (Monarch)                   |
            | **WPO**              | *C. elegans* Phenotype Ontology (Monarch)        |
            | **ZPO**              | Zebrafish Phenotype Ontology (Monarch)           |
            | **FYPO**             | Fission Yeast Phenotype Ontology (Monarch)       |
            | **Pfam**             | Protein Domains (Pfam)                           |
            | **SMART**            | Protein Domains (SMART)                          |
            | **InterPro**         | Protein Domains and Features (InterPro)          |
            | **PMID**             | Reference Publications (PubMed)                  |
            | **NetworkNeighborAL**| Local Network Cluster (STRING)                   |
            | **COMPARTMENTS**     | Subcellular Localization (COMPARTMENTS)          |
            | **TISSUES**          | Tissue Expression (TISSUES)                      |
            | **DISEASES**         | Disease–gene Associations (DISEASES)             |
            | **WikiPathways**     | WikiPathways                                     |

        Note:
            The `key` must correspond to an existing entry in `.stats["functional"]`, created via 
            `enrichment_functional()`.
        """
        from xml.parsers.expat import ExpatError

        if "functional" not in self.stats:
            raise ValueError("No STRING enrichment results found in .stats['functional'].")

        all_keys = list(self.stats["functional"].keys())

        # Handle DE-type key
        if "vs" in key:
            if direction not in {"up", "down"}:
                raise ValueError("You must specify direction='up' or 'down' for DE-based enrichment keys.")
            lookup_key = _resolve_de_key(self.stats["functional"], f"{key}_{direction}")
        else:
            # Handle user-supplied key (e.g. "userSearch1")
            if direction is not None:
                print(f"[WARNING] Ignoring direction='{direction}' for user-supplied key: '{key}'")
            lookup_key = key

        if lookup_key not in self.stats["functional"]:
            available = "\n".join(f"  - {k}" for k in self.stats["functional"].keys())
            raise ValueError(f"Could not find enrichment results for '{lookup_key}'. Available keys:\n{available}")

        meta = self.stats["functional"][lookup_key]
        string_ids = meta["string_ids"]
        species_id = meta["species"]

        url = "https://string-db.org/api/svg/enrichmentfigure"
        params = {
            "identifiers": "%0d".join(string_ids),
            "species": species_id
        }
        if category:
            params["category"] = category

        print(f"{format_log_prefix('user')} Fetching STRING SVG for key '{lookup_key}' (n={len(string_ids)})...")
        response = requests.get(url, params=params)
        response.raise_for_status()

        if save_as:
            with open(save_as, "wb") as f:
                f.write(response.content)
            print(f"{format_log_prefix('info_only')} Saved SVG to: {save_as}")

        with tempfile.NamedTemporaryFile("wb", suffix=".svg", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        try:
            try:
                display(SVG(filename=tmp_path))
            except ExpatError:
                print(
                    f"{format_log_prefix('info_only')} No enrichment figure available "
                    f"for key '{lookup_key}'"
                    + (f" (category='{category}')." if category else ".")
                )
        finally:
            os.remove(tmp_path)

    def get_string_network_link(self, key=None, string_ids=None, species=None, show_labels=True):
        """
        Generate a direct STRING network URL to visualize protein interactions online.

        This method constructs a STRING website link to view a network of proteins,
        using either a list of STRING IDs or a key from previously stored enrichment results.

        Args:
            key (str, optional): Key from `.stats["functional"]` to extract STRING IDs and species info.
            string_ids (list of str, optional): List of STRING identifiers to include in the network.
            species (int or str, optional): NCBI taxonomy ID (e.g., 9606 for human). Required if not using a stored key.
            show_labels (bool): If True (default), node labels will be shown in the network view.

        Returns:
            str: URL to open the network in the STRING web interface.

        Example:
            Get a STRING network link for a stored enrichment result:
                ```python
                url = pdata.get_string_network_link(key="UserSearch1")
                print(url)
                ```
        """
        if string_ids is None:
            if key is None:
                raise ValueError("Must provide either a list of STRING IDs or a key.")
            metadata = self.stats.get("functional", {}).get(key)
            if metadata is None:
                raise ValueError(f"Key '{key}' not found in self.stats['functional'].")
            string_ids = metadata.get("string_ids")
            species = species or metadata.get("species")

        if not string_ids:
            raise ValueError("No STRING IDs found or provided.")

        base_url = "https://string-db.org/cgi/network"
        params = [
            f"identifiers={'%0d'.join(string_ids)}",
            f"caller_identity=scpviz"
        ]
        if species:
            params.append(f"species={species}")
        if show_labels:
            params.append("show_query_node_labels=1")

        return f"{base_url}?{'&'.join(params)}"

# --- Top-level helper (keep above class definition) ---
def _pretty_vs_key(k):
    """
    Format a DE contrast key into a human-readable string.

    This function attempts to convert a string representation of a DE comparison
    (e.g., a list of dictionaries) into a simplified `"group1 vs group2"` format,
    using the values from each dictionary in the left and right group.

    Args:
        k (str): DE key string, typically in the format `"[{{...}}] vs [{{...}}]"`.

    Returns:
        str: A simplified, human-readable version of the DE comparison key.
    """
    import ast
    try:
        parts = k.split(" vs ")
        left = "_".join(str(v) for d in ast.literal_eval(parts[0]) for v in d.values())
        right = "_".join(str(v) for d in ast.literal_eval(parts[1]) for v in d.values())
        return f"{left} vs {right}"
    except Exception:
        return k  # fallback to raw key if anything goes wrong

def _resolve_de_key(stats_dict, user_key, debug=False):
    """
    Resolve a user-supplied DE key to a valid key stored in `.stats["de_results"]`.

    This function matches a flexible, human-readable DE key against the internal keys
    stored in the DE results dictionary. It supports both raw and pretty-formatted keys,
    and can handle suffixes like `_up` or `_down` for directional analysis.

    Args:
        stats_dict (dict): Dictionary of DE results (typically `pdata.stats["de_results"]`).
        user_key (str): User-supplied key to resolve, e.g., "AS_kd vs AS_sc_down".
        debug (bool): If True, prints detailed debug output for tracing.

    Returns:
        str: The matching internal DE result key.

    Raises:
        ValueError: If no matching key is found.
    """
    import re
    print(f"[DEBUG] Resolving user key: {user_key}") if debug else None

    # Extract suffix
    suffix = ""
    if user_key.endswith("_up") or user_key.endswith("_down"):
        user_key, suffix = re.match(r"(.+)(_up|_down)", user_key).groups()
        print(f"[DEBUG] Split into base='{user_key}', suffix='{suffix}'") if debug else None

    # Build pretty key mapping
    pretty_map = {}
    for full_key in stats_dict:
        if "vs" not in full_key:
            continue

        if full_key.endswith("_up") or full_key.endswith("_down"):
            base = full_key.rsplit("_", 1)[0]
            full_suffix = "_" + full_key.rsplit("_", 1)[1]
        else:
            base = full_key
            full_suffix = ""

        pretty_key = _pretty_vs_key(base)
        final_key = pretty_key + full_suffix
        pretty_map[final_key] = full_key
        print(f"[DEBUG] Mapped '{final_key}' → '{full_key}'")  if debug else None

    full_user_key = user_key + suffix
    print(f"[DEBUG] Full user key for lookup: '{full_user_key}'")  if debug else None

    # Also try pretty-formatting the *user* key so legacy dict-style keys
    # (e.g. "[{'cellline': 'BE'}] vs …") resolve against modern stats keys
    # like "BE_kd vs BE_sc".
    pretty_user_key = _pretty_vs_key(user_key) + suffix
    print(f"[DEBUG] Pretty user key for lookup: '{pretty_user_key}'") if debug else None

    if full_user_key in stats_dict:
        print("[DEBUG] Found direct match in stats.") if debug else None
        return full_user_key
    if full_user_key in pretty_map:
        print(f"[DEBUG] Found in pretty map: {pretty_map[full_user_key]}")  if debug else None
        return pretty_map[full_user_key]
    if pretty_user_key != full_user_key and pretty_user_key in stats_dict:
        print("[DEBUG] Found pretty-formatted user key in stats.") if debug else None
        return pretty_user_key
    if pretty_user_key != full_user_key and pretty_user_key in pretty_map:
        print(f"[DEBUG] Found pretty user key in pretty map: {pretty_map[pretty_user_key]}") if debug else None
        return pretty_map[pretty_user_key]

    pretty_keys = "\n".join(f"  - {k}" for k in pretty_map.keys()) if pretty_map else "  (none found)"
    raise ValueError(
        f"'{full_user_key}' not found in stats.\n"
        f"Available DE keys:\n{pretty_keys}"
    )

# --- re-export methods to top level for testing convenience ---
def enrichment_functional(pdata, *args, **kwargs):
    return EnrichmentMixin.enrichment_functional(pdata, *args, **kwargs)

def enrichment_ppi(pdata, *args, **kwargs):
    return EnrichmentMixin.enrichment_ppi(pdata, *args, **kwargs)

def enrichment_functional_pca(pdata, *args, **kwargs):
    return EnrichmentMixin.enrichment_functional_pca(pdata, *args, **kwargs)
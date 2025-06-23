import requests
import pandas as pd
import time
from io import StringIO
from IPython.display import SVG, display
import tempfile
import os

from scviz.utils import format_log_prefix

class EnrichmentMixin:
    """
    Performs functional enrichment and PPI analysis via STRING.

    Functions:
        get_string_mappings: Map protein IDs to STRING IDs.
        resolve_to_accessions: Convert gene names to protein accessions.
        enrichment_functional: Functional enrichment (GO, KEGG, etc.).
        enrichment_ppi: Protein-protein interaction network enrichment.
        list_enrichments: View stored enrichment results.
        plot_enrichment_svg: Render STRING SVG network plots.
        get_string_network_link: Generate STRING network visualization URLs.
    """        
    def get_string_mappings(self, identifiers, batch_size=300, debug=False):
        """
        Maps UniProt accessions to STRING IDs, with caching into self.prot.var["String ID"].

        Parameters
        ----------
        identifiers : list of str
            List of UniProt accessions.
        batch_size : int
            Batch size for querying STRING API.

        Returns
        -------
        pd.DataFrame
            Mapping table with columns: input_identifier, string_identifier, ncbi_taxon_id
        """
        print(f"[INFO] Resolving STRING IDs for {len(identifiers)} identifiers...") if debug else None

        prot_var = self.prot.var
        if "String ID" not in prot_var.columns:
            prot_var["String ID"] = pd.NA

        # Use cached STRING IDs if available
        valid_ids = [i for i in identifiers if i in prot_var.index]
        existing = prot_var.loc[valid_ids, "String ID"]
        found_ids = {i: sid for i, sid in existing.items() if pd.notna(sid)}
        missing = [i for i in identifiers if i not in found_ids]

        print(f"{format_log_prefix('info_only',2)} Found {len(found_ids)} cached STRING IDs. {len(missing)} need lookup.")

        all_rows = []

        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            print(f"{format_log_prefix('info')} Querying STRING for batch {i // batch_size + 1} ({len(batch)} identifiers)...") if debug else None

            url = "https://version-12-0.string-db.org/api/tsv-no-header/get_string_ids"
            params = {
                "identifiers": "\r".join(batch),
                "limit": 1,
                "echo_query": 1,
                "caller_identity": "scviz"
            }

            try:
                t0 = time.time()
                response = requests.post(url, data=params)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text), sep="\t", header=None)
                df.columns = [
                    "input_identifier", "input_alias", "string_identifier", "ncbi_taxon_id",
                    "preferred_name", "annotation", "score"
                ]
                print(f"[INFO] Batch completed in {time.time() - t0:.2f}s") if debug else None
                all_rows.append(df)
            except Exception as e:
                print(f"[ERROR] Failed on batch {i // batch_size + 1}: {e}")

        # Combine all new mappings
        if all_rows:
            new_df = pd.concat(all_rows, ignore_index=True)
            updated_ids = []

            for _, row in new_df.iterrows():
                acc = row["input_identifier"]
                sid = row["string_identifier"]
                if acc in self.prot.var.index:
                    self.prot.var.at[acc, "String ID"] = sid
                    found_ids[acc] = sid
                    updated_ids.append(acc)
                else:
                    print(f"[DEBUG] Skipping unknown accession '{acc}'")

            print(f"{format_log_prefix('info')} Cached {len(updated_ids)} new STRING ID mappings.")
        else:
            print(f"{format_log_prefix('warn')} No STRING mappings returned for the requested identifiers.")

        # Return final mapping as a DataFrame
        out_df = pd.DataFrame.from_dict(found_ids, orient="index", columns=["string_identifier"])
        out_df.index.name = "input_identifier"
        out_df = out_df.reset_index()
        out_df["ncbi_taxon_id"] = 9606  # default fallback if needed
        return out_df

    def resolve_to_accessions(self, mixed_list):
        """
        Converts a list of gene names or accessions into accessions
        using internal gene-to-accession mapping.

        Parameters
        ----------
        mixed_list : list of str
            Gene names or UniProt accessions.

        Returns
        -------
        list of str
            Accessions resolved from input items.
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
        top_n=150,
        score_col="significance_score",
        gene_col="Genes",
        de_key="de_results",
        store_key=None,
        species=None,
        background=None,
        debug=False,
        **kwargs
    ):

        def query_functional_enrichment(query_ids, species_id, background_ids=None, debug=False):
            # print(f"{format_log_prefix('info_only',2)} Running enrichment on {len(query_ids)} STRING IDs (species {species_id})...")
            url = "https://version-12-0.string-db.org/api/json/enrichment"
            payload = {
                "identifiers": "%0d".join(query_ids),
                "species": species_id,
                "caller_identity": "scviz"
            }
            if background_ids is not None:
                print(f"{format_log_prefix('info_only')} Using background of {len(background_ids)} STRING IDs.")
                payload["background_string_identifiers"] = "%0d".join(background_ids)

            response = requests.post(url, data=payload)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        
        # Ensure string metadata section exists
        if "functional" not in self.stats:
            self.stats["functional"] = {}

        if genes is None and from_de:
            resolved_key = _resolve_de_key(self.stats, de_key)
            de_df = self.stats[resolved_key]
            sig_df = de_df[de_df["significance"] != "not significant"].copy()
            print(f"{format_log_prefix('user')} Running STRING enrichment [DE-based: {resolved_key}]")

            up_genes = sig_df[sig_df[score_col] > 0][gene_col].dropna().head(top_n).tolist()
            down_genes = sig_df[sig_df[score_col] < 0][gene_col].dropna().head(top_n).tolist()

            up_accs, up_unresolved = self.resolve_to_accessions(up_genes)
            down_accs, down_unresolved = self.resolve_to_accessions(down_genes)

            background_accs = None
            background_string_ids = None
            if background == "all_quantified":
                print(f"{format_log_prefix('warn')} Mapping background proteins may take a long time due to batching.")
                background_accs = de_df[de_df["significance"] == "not significant"].index.tolist()

            if background_accs:
                bg_map = self.get_string_mappings(background_accs)
                background_string_ids = bg_map["string_identifier"].tolist()

            if store_key is not None:
                print(f"{format_log_prefix('warn')} Ignoring `store_key` for DE-based enrichment. Using auto-generated pretty keys.")

            results = {}
            for label, accs in zip(["up", "down"], [up_accs, down_accs]):
                t0 = time.time()

                if not accs:
                    print(f"{format_log_prefix('warn')} No {label}-regulated proteins to analyze.")
                    continue

                mapping_df = self.get_string_mappings(accs)
                if mapping_df.empty:
                    print(f"{format_log_prefix('warn')} No valid STRING mappings found for {label}-regulated proteins.")
                    continue

                string_ids = mapping_df["string_identifier"].tolist()
                inferred_species = mapping_df["ncbi_taxon_id"].mode().iloc[0]
                species_id = species if species is not None else inferred_species

                print(f"\n🔹 {label.capitalize()}-regulated proteins")
                print(f"   🔸 Proteins: {len(accs)} → STRING IDs: {len(string_ids)}")
                print(f"   🔸 Species: {species_id} | Background: {'None' if background_string_ids is None else 'custom'}")
                if label == "up":
                    if up_unresolved:
                        print(f"{format_log_prefix('warn',2)} Some accessions unresolved for {label}-regulated proteins: {', '.join(up_unresolved)}")
                else:
                    if down_unresolved:
                        print(f"{format_log_prefix('warn',2)} Some accessions unresolved for {label}-regulated proteins: {', '.join(down_unresolved)}")

                enrichment_df = query_functional_enrichment(string_ids, species_id, background_string_ids)
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
            mapping_df = self.get_string_mappings(input_accs)

            if mapping_df.empty:
                raise ValueError("No valid STRING mappings found for the provided identifiers.")

            string_ids = mapping_df["string_identifier"].tolist()
            inferred_species = mapping_df["ncbi_taxon_id"].mode().iloc[0]
            species_id = species if species is not None else inferred_species

            background_string_ids = None
            if background == "all_quantified":
                print(f"{format_log_prefix('warn')} Mapping background proteins may take a long time due to batching.")
                all_accs = list(self.prot.var_names)
                background_accs = list(set(all_accs) - set(input_accs))
                bg_map = self.get_string_mappings(background_accs)
                background_string_ids = bg_map["string_identifier"].tolist()

            print(f"   🔸 Input genes: {len(genes)} → Resolved STRING IDs: {len(string_ids)}")
            print(f"   🔸 Species: {species_id} | Background: {'None' if background_string_ids is None else 'custom'}")
            if unresolved_accs:
                print(f"{format_log_prefix('warn',2)} Some accessions unresolved: {', '.join(unresolved_accs)}")

            enrichment_df = query_functional_enrichment(string_ids, species_id, background_string_ids)
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

    def enrichment_ppi(self, genes, species=None, store_key=None):
        """
        Run STRING PPI enrichment on a user-supplied list of accessions or gene names.

        Parameters
        ----------
        genes : list of str
            A list of accessions or gene names.
        species : int or None
            NCBI species ID (e.g. 9606 for human). If None, inferred from STRING ID mapping.
        store_key : str or None
            Key to store the PPI enrichment result in self.stats["ppi"]. If None, auto-generates a unique key.
        """
        def query_ppi_enrichment(string_ids, species):
            # print(f"[INFO] Running PPI enrichment for {len(string_ids)} STRING IDs (species {species})...")
            url = "https://string-db.org/api/json/ppi_enrichment"
            payload = {
                "identifiers": "%0d".join(string_ids),
                "species": species,
                "caller_identity": "scviz"
            }

            response = requests.post(url, data=payload)
            response.raise_for_status()

            result = response.json()
            print("[DEBUG] PPI enrichment result:", result)
            return result[0] if isinstance(result, list) else result

        print(f"{format_log_prefix('user')} Running STRING PPI enrichment")
        t0 = time.time()
        input_accs, unresolved_accs = self.resolve_to_accessions(genes)
        mapping_df = self.get_string_mappings(input_accs)

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

    def list_enrichments(self):
        """
        List available STRING enrichment results and DE contrasts not yet analyzed.
        Always outputs as plain text (for use in scripts, terminals, notebooks).
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
        Display STRING enrichment SVG inline in Jupyter.

        Parameters
        ----------
        key : str
            Key to use from .stats["functional"], e.g. a DE contrast or 'userSearch1'.
        direction : str or None
            'up' or 'down' for DE comparisons. Should be None for user-supplied lists.
        category : str or None
            STRING enrichment category ("GO", "KEGG", etc).
        save_as : str or None
            If provided, also saves the SVG to this path.
        """
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
            display(SVG(filename=tmp_path))
        finally:
            os.remove(tmp_path)

    def get_string_network_link(self, key=None, string_ids=None, species=None, show_labels=True):
        """
        Generate a direct STRING website link to view a network of the given STRING IDs.

        If `key` is provided, retrieves from self.stats["functional"][key].

        Returns
        -------
        str : URL
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
            f"caller_identity=scviz"
        ]
        if species:
            params.append(f"species={species}")
        if show_labels:
            params.append("show_query_node_labels=1")

        return f"{base_url}?{'&'.join(params)}"

# --- Top-level helper (keep above class definition) ---
def _pretty_vs_key(k):
    import ast
    try:
        parts = k.split(" vs ")
        left = "_".join(str(v) for d in ast.literal_eval(parts[0]) for v in d.values())
        right = "_".join(str(v) for d in ast.literal_eval(parts[1]) for v in d.values())
        return f"{left} vs {right}"
    except Exception:
        return k  # fallback to raw key if anything goes wrong

def _resolve_de_key(stats_dict, user_key, debug=False):
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

    if full_user_key in stats_dict:
        print("[DEBUG] Found direct match in stats.") if debug else None
        return full_user_key
    elif full_user_key in pretty_map:
        print(f"[DEBUG] Found in pretty map: {pretty_map[full_user_key]}")  if debug else None
        return pretty_map[full_user_key]
    else:
        pretty_keys = "\n".join(f"  - {k}" for k in pretty_map.keys()) if pretty_map else "  (none found)"
        raise ValueError(
            f"'{full_user_key}' not found in stats.\n"
            f"Available DE keys:\n{pretty_keys}"
        )

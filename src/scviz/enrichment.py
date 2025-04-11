import requests
import pandas as pd
import time
from io import StringIO
import ast
from IPython.display import SVG, display
import tempfile
import os

def _pretty_vs_key(k):
    import ast
    try:
        parts = k.split(" vs ")
        left = "_".join(str(v) for d in ast.literal_eval(parts[0]) for v in d.values())
        right = "_".join(str(v) for d in ast.literal_eval(parts[1]) for v in d.values())
        return f"{left} vs {right}"
    except Exception:
        return k  # fallback to raw key if anything goes wrong

def _resolve_de_key(stats_dict, user_key):
    import re
    print(f"[DEBUG] Resolving user key: {user_key}")

    # Extract suffix
    suffix = ""
    if user_key.endswith("_up") or user_key.endswith("_down"):
        user_key, suffix = re.match(r"(.+)(_up|_down)", user_key).groups()
        print(f"[DEBUG] Split into base='{user_key}', suffix='{suffix}'")

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
        print(f"[DEBUG] Mapped '{final_key}' → '{full_key}'")

    full_user_key = user_key + suffix
    print(f"[DEBUG] Full user key for lookup: '{full_user_key}'")

    if full_user_key in stats_dict:
        print("[DEBUG] Found direct match in stats.")
        return full_user_key
    elif full_user_key in pretty_map:
        print(f"[DEBUG] Found in pretty map: {pretty_map[full_user_key]}")
        return pretty_map[full_user_key]
    else:
        pretty_keys = "\n".join(f"  - {k}" for k in pretty_map.keys()) if pretty_map else "  (none found)"
        raise ValueError(
            f"'{full_user_key}' not found in stats.\n"
            f"Available DE keys:\n{pretty_keys}"
        )


def gsea_analysis(
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
    **kwargs
):
    def resolve_to_accessions(mixed_list):
        gene_to_acc, _ = self.get_gene_maps(on='protein')
        accs = []
        for item in mixed_list:
            if item in self.prot.var.index:
                accs.append(item)  # already an accession
            elif item in gene_to_acc:
                accs.append(gene_to_acc[item])
            else:
                print(f"[WARNING] Could not resolve '{item}' to an accession — skipping.")
        return accs

    def get_string_mappings(identifiers, batch_size=300):
        print(f"[INFO] Resolving STRING IDs for {len(identifiers)} identifiers...")

        prot_var = self.prot.var
        if "String ID" not in prot_var.columns:
            prot_var["String ID"] = pd.NA

        # Use cached STRING IDs if available
        valid_ids = [i for i in identifiers if i in prot_var.index]
        existing = prot_var.loc[valid_ids, "String ID"]
        found_ids = {i: sid for i, sid in existing.items() if pd.notna(sid)}
        missing = [i for i in identifiers if i not in found_ids]

        print(f"[INFO] Found {len(found_ids)} cached STRING IDs. {len(missing)} need lookup.")

        all_rows = []

        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            print(f"[INFO] Querying STRING for batch {i // batch_size + 1} ({len(batch)} identifiers)...")

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
                print(f"[INFO] Batch completed in {time.time() - t0:.2f}s")
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

            print(f"[INFO] Cached {len(updated_ids)} new STRING ID mappings.")
        else:
            print("[WARNING] No STRING mappings returned for the requested identifiers.")

        # Return final mapping as a DataFrame
        out_df = pd.DataFrame.from_dict(found_ids, orient="index", columns=["string_identifier"])
        out_df.index.name = "input_identifier"
        out_df = out_df.reset_index()
        out_df["ncbi_taxon_id"] = 9606  # default fallback if needed
        return out_df

    def query_string_enrichment(query_ids, species_id, background_ids=None):
        print(f"[INFO] Running enrichment on {len(query_ids)} STRING IDs (species {species_id})...")
        url = "https://version-12-0.string-db.org/api/json/enrichment"
        payload = {
            "identifiers": "%0d".join(query_ids),
            "species": species_id,
            "caller_identity": "scviz"
        }
        if background_ids is not None:
            print(f"[INFO] Using background of {len(background_ids)} STRING IDs.")
            payload["background_string_identifiers"] = "%0d".join(background_ids)

        t0 = time.time()
        response = requests.post(url, data=payload)
        response.raise_for_status()
        print(f"[INFO] Enrichment completed in {time.time() - t0:.2f}s")
        return pd.DataFrame(response.json())
    
    # Ensure string metadata section exists
    if "string" not in self.stats:
        self.stats["string"] = {}

    if genes is None and from_de:
        resolved_key = _resolve_de_key(self.stats, de_key)
        de_df = self.stats[resolved_key]
        sig_df = de_df[de_df["significance"] != "not significant"].copy()

        up_genes = sig_df[sig_df[score_col] > 0][gene_col].dropna().head(top_n).tolist()
        down_genes = sig_df[sig_df[score_col] < 0][gene_col].dropna().head(top_n).tolist()

        up_accs = resolve_to_accessions(up_genes)
        down_accs = resolve_to_accessions(down_genes)

        background_accs = None
        background_string_ids = None
        if background == "all_quantified":
            print("[WARNING] Mapping background proteins may take a long time due to batching.")
            background_accs = de_df[de_df["significance"] == "not significant"].index.tolist()

        if background_accs:
            bg_map = get_string_mappings(background_accs)
            background_string_ids = bg_map["string_identifier"].tolist()

        results = {}
        for label, accs in zip(["up", "down"], [up_accs, down_accs]):
            if not accs:
                print(f"[WARNING] No {label}-regulated proteins to analyze.")
                continue

            mapping_df = get_string_mappings(accs)
            if mapping_df.empty:
                print(f"[WARNING] No valid STRING mappings found for {label}-regulated proteins.")
                continue

            string_ids = mapping_df["string_identifier"].tolist()
            inferred_species = mapping_df["ncbi_taxon_id"].mode().iloc[0]
            species_id = species if species is not None else inferred_species

            enrichment_df = query_string_enrichment(string_ids, species_id, background_string_ids)
            enrich_key = f"{resolved_key}_{label}"
            self.stats[f"{store_key}_{label}"] = enrichment_df
            self.stats["string"][enrich_key] = {
                "string_ids": string_ids,
                "background_string_ids": background_string_ids,
                "species": species_id
            }
            results[label] = enrichment_df

            pretty_base = _pretty_vs_key(resolved_key)
            pretty_key = f"{pretty_base}_{label}"
            print(f"[INFO] Stored enrichment and metadata under keys:")
            print(f"  - {pretty_key}_up")
            print(f"  - {pretty_key}_down")
            print(f"[INFO] Access enrichment table: pdata.stats[\"{enrich_key}\"]")
            print(f"[INFO] To visualize in notebook: pdata.plot_enrichment_svg(\"{pretty_base}\", direction=\"{label}\")")

    elif genes is not None:
        if store_key is None:
            prefix = "UserSearch"
            existing = self.stats["string"].keys() if "string" in self.stats else []
            existing_ids = [k for k in existing if k.startswith(prefix)]
            next_id = len(existing_ids) + 1
            store_key = f"{prefix}{next_id}"

        input_accs = resolve_to_accessions(genes)
        mapping_df = get_string_mappings(input_accs)

        if mapping_df.empty:
            raise ValueError("No valid STRING mappings found for the provided identifiers.")

        string_ids = mapping_df["string_identifier"].tolist()
        inferred_species = mapping_df["ncbi_taxon_id"].mode().iloc[0]
        species_id = species if species is not None else inferred_species

        background_string_ids = None
        if background == "all_quantified":
            print("[WARNING] Mapping background proteins may take a long time due to batching.")
            all_accs = list(self.prot.var_names)
            background_accs = list(set(all_accs) - set(input_accs))
            bg_map = get_string_mappings(background_accs)
            background_string_ids = bg_map["string_identifier"].tolist()

        enrichment_df = query_string_enrichment(string_ids, species_id, background_string_ids)
        self.stats[store_key] = enrichment_df
        self.stats["string"][store_key] = {
            "string_ids": string_ids,
            "background_string_ids": background_string_ids,
            "species": species_id
        }

        print(f"[INFO] Stored enrichment and metadata under key: {store_key}")
        print(f"[INFO] Access enrichment table: pdata.stats[\"{store_key}\"]")
        print(f"[INFO] To visualize in notebook: pdata.plot_enrichment_svg(\"{store_key}\")")

        return enrichment_df

    else:
        raise ValueError("Must provide 'genes' or set from_de=True to use DE results.") 



def plot_enrichment_svg(self, key, direction=None, category=None, save_as=None):
    """
    Display STRING enrichment SVG inline in Jupyter.

    Parameters
    ----------
    key : str
        Key to use from .stats["string"], e.g. a DE contrast or 'userSearch1'.
    direction : str or None
        'up' or 'down' for DE comparisons. Should be None for user-supplied lists.
    category : str or None
        STRING enrichment category ("GO", "KEGG", etc).
    save_as : str or None
        If provided, also saves the SVG to this path.
    """
    if "string" not in self.stats:
        raise ValueError("No STRING enrichment results found in .stats['string'].")

    all_keys = list(self.stats["string"].keys())

    # Handle DE-type key
    if "vs" in key:
        if direction not in {"up", "down"}:
            raise ValueError("You must specify direction='up' or 'down' for DE-based enrichment keys.")
        lookup_key = _resolve_de_key(self.stats["string"], f"{key}_{direction}")
    else:
        # Handle user-supplied key (e.g. "userSearch1")
        if direction is not None:
            print(f"[WARNING] Ignoring direction='{direction}' for user-supplied key: '{key}'")
        lookup_key = key

    if lookup_key not in self.stats["string"]:
        available = "\n".join(f"  - {k}" for k in self.stats["string"].keys())
        raise ValueError(f"Could not find enrichment results for '{lookup_key}'. Available keys:\n{available}")

    meta = self.stats["string"][lookup_key]
    string_ids = meta["string_ids"]
    species_id = meta["species"]

    url = "https://string-db.org/api/svg/enrichmentfigure"
    params = {
        "identifiers": "%0d".join(string_ids),
        "species": species_id
    }
    if category:
        params["category"] = category

    print(f"[INFO] Fetching STRING SVG for key '{lookup_key}' (n={len(string_ids)})...")
    response = requests.get(url, params=params)
    response.raise_for_status()

    if save_as:
        with open(save_as, "wb") as f:
            f.write(response.content)
        print(f"[INFO] Saved SVG to: {save_as}")

    with tempfile.NamedTemporaryFile("wb", suffix=".svg", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        display(SVG(filename=tmp_path))
    finally:
        os.remove(tmp_path)

def list_enrichments(self):
    """
    List available STRING enrichment results and DE contrasts not yet analyzed.
    Always outputs as plain text (for use in scripts, terminals, notebooks).
    """

    string_keys = set(self.stats.get("string", {}).keys())
    all_stats_keys = set(self.stats.keys())
    de_keys = {k for k in all_stats_keys if "vs" in k}

    # Already enriched DE keys (up/down)
    enriched_de = set()
    for k in string_keys:
        if "vs" in k and (k.endswith("_up") or k.endswith("_down")):
            base = k.rsplit("_", 1)[0]
            enriched_de.add(base)

    # DE keys with no enrichment yet
    de_unenriched = de_keys - enriched_de
    unenriched_pretty = sorted(_pretty_vs_key(k) for k in de_unenriched)

    enriched_results = []
    for k in sorted(string_keys):
        if "vs" in k:
            if k.endswith("_up") or k.endswith("_down"):
                base, suffix = k.rsplit("_", 1)
                pretty = f"{_pretty_vs_key(base)}_{suffix}"
            else:
                pretty = _pretty_vs_key(k)
            entry_type = "DE-based"
        else:
            pretty = k
            entry_type = "User"
        enriched_results.append((pretty, k, entry_type))

    print("\n[STRING Enrichment Summary]\n")

    print("Available DE comparisons (not yet enriched):")
    if unenriched_pretty:
        for pk in unenriched_pretty:
            print(f"  - {pk}")
        print('To run enrichment: pdata.gsea_analysis(from_de=True, de_key="...")\n')
    else:
        print("  (none)\n")

    print("Completed STRING enrichment results:")
    if not enriched_results:
        print("  (none)")
    for pretty, raw_key, kind in enriched_results:
        if kind == "DE-based":
            base, suffix = pretty.rsplit("_", 1)
            print(f"  - {pretty} ({kind})")
            print(f"    Access table: pdata.stats[\"{raw_key}\"]")
            print(f"    Plot result : pdata.plot_enrichment_svg(\"{base}\", direction=\"{suffix}\")\n")
        else:
            print(f"  - {pretty} ({kind})")
            print(f"    Access table: pdata.stats[\"{raw_key}\"]")
            print(f"    Plot result : pdata.plot_enrichment_svg(\"{pretty}\")\n")

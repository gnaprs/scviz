from tkinter import N
import numpy as np
import pandas as pd
from scpviz import utils
import re

from scpviz.utils import format_log_prefix
import warnings

def _detect_ambiguous_input(group, var, group_metrics=None):
    """
    Detects ambiguous user input mixing file and group identifiers.

    This helper checks whether the `group` list includes both
    file-like identifiers (present in `.var` as 'Found In: <file>')
    and group-like identifiers (present in `.uns` as ('group', 'count') tuples).

    Args:
        group (list of str): User-provided identifiers for filtering.
        var (pd.DataFrame): The `.var` table of the corresponding AnnData object.
        group_metrics (pd.DataFrame, optional): MultiIndex DataFrame from `.uns`
            containing per-group ('count', 'ratio') metrics.

    Returns:
        tuple(bool, list, list):
            (is_ambiguous, annotated_files, annotated_groups)
            - is_ambiguous (bool): True if both file-like and group-like entries coexist.
            - annotated_files (list): Entries that match file-level columns.
            - annotated_groups (list): Entries that match group-level metrics.
    """
    annotated_files = [g for g in group if f"Found In: {g}" in var.columns]
    annotated_groups = []
    if group_metrics is not None:
        annotated_groups = [
            g for g in group
            if (g, "count") in group_metrics.columns or (g, "ratio") in group_metrics.columns
        ]

    has_file_like = bool(annotated_files)
    has_group_like = bool(annotated_groups)

    # Ambiguous only if both exist and sets are distinct
    is_ambiguous = has_file_like and has_group_like and set(annotated_files) != set(annotated_groups)

    return is_ambiguous, annotated_files, annotated_groups

class FilterMixin:
    """
    Provides flexible filtering and annotation methods for samples, proteins, and peptides.

    This mixin includes utilities for:

    - Filtering proteins and peptides by metadata conditions, group-level detection, or peptide mapping structure.
    - Filtering samples based on class annotations, numeric thresholds, file lists, or query strings.
    - Annotating detection status ("Found In") across samples and class-based groups.
    - Managing and validating the protein–peptide relational structure (RS matrix) after filtering.

    Functions:
        filter_prot: Filters proteins using `.var` metadata conditions or a list of accessions/genes to retain.
        filter_prot_found: Keeps proteins or peptides found in a minimum number or proportion of samples within a group or file list.
        _filter_sync_peptides_to_proteins: Removes peptides orphaned by upstream protein filtering.
        filter_sample: Filters samples using categorical metadata, numeric thresholds, or file/sample lists.
        _filter_sample_condition: Internal helper for filtering samples using `.summary` conditions or name lists.
        _filter_sample_values: Filters samples using dictionary-style matching on metadata fields.
        _filter_sample_query: Parses and applies a raw pandas-style query string to `.obs` or `.summary`.
        filter_rs: Filters the RS matrix by peptide count and ambiguity, and updates `.prot`/`.pep` accordingly.
        _apply_rs_filter: Applies protein/peptide masks to `.prot`, `.pep`, and `.rs` matrices.
        _format_filter_query: Formats filter conditions for `.eval()` by quoting fields and handling `includes` syntax.
        annotate_found: Adds group-level "Found In" indicators to `.prot.var` or `.pep.var`.
        _annotate_found_samples: Computes per-sample detection flags for use by `annotate_found()`.
    """
    
    def filter_prot(self, condition = None, accessions=None, valid_genes=False, unique_profiles=False, return_copy = True, debug=False):
        """
        Filter protein data based on metadata conditions or accession list (protein name and gene name).

        This method filters the protein-level data either by evaluating a string condition on the protein metadata,
        or by providing a list of protein accession numbers (or gene names) to keep. Peptides that are exclusively
        linked to removed proteins are also removed, and the RS matrix is updated accordingly.
        
        Args:
            condition (str): A condition string to filter protein metadata. Supports:

                - Standard comparisons, e.g. `"Protein FDR Confidence: Combined == 'High'"`
                - Substring queries using `includes`, e.g. `"Description includes 'p97'"`
            accessions (list of str, optional): List of accession numbers (var_names) to keep.
            valid_genes (bool): If True, removes rows with missing gene names and resolves duplicate gene names by appending numeric suffixes.
            unique_profiles (bool): If True, remove rows with duplicate abundance profiles across samples.
            return_copy (bool): If True, returns a filtered copy. If False, modifies in place.
            debug (bool): If True, prints debugging information.

        Returns:
            pAnnData (pAnnData): Returns a filtered pAnnData object if `return_copy=True`. 
            None (None): Otherwise, modifies in-place and returns None.

        Examples:
            Filter by metadata condition:
                ```python
                condition = "Protein FDR Confidence: Combined == 'High'"
                pdata.filter_prot(condition=condition)
                ```

            Substring match on protein description:
                ```python
                condition = "Description includes 'p97'"
                pdata.filter_prot(condition=condition)
                ```

            Numerical condition on metadata:
                ```python
                condition = "Score > 0.75"
                pdata.filter_prot(condition=condition)
                ```

            Filter by specific protein accessions:
                ```python
                accessions = ['GAPDH', 'P53']
                pdata.filter_prot(accessions=accessions)
                ```

            Filter out all that have no valid genes (potentially artefacts):
                ```python
                pdata.filter_prot(valid_genes=True)
                ```

            !!! tip
                Multiple filters can be combined in a single call. For example, to filger by condition and valid genes:
                ```python
                condition = "Score > 0.75"
                pdata.filter_prot(condition=condition, valid_genes=True)
        """
        from scipy.sparse import issparse

        if not self._check_data('protein'): # type: ignore[attr-defined]
            raise ValueError(f"No protein data found. Check that protein data was imported.")

        pdata = self.copy() if return_copy else self # type: ignore[attr-defined]
        action = "Returning a copy of" if return_copy else "Filtered and modified"

        message_parts = []

        # 1. Filter by condition
        if condition is not None:
            formatted_condition = self._format_filter_query(condition, pdata.prot.var)
            if debug:
                print(f"Formatted condition: {formatted_condition}")
            filtered_proteins = pdata.prot.var[pdata.prot.var.eval(formatted_condition)]
            pdata.prot = pdata.prot[:, filtered_proteins.index]
            message_parts.append(f"condition: {condition}")

        # 2. Filter by accession list or gene names
        if accessions is not None:
            gene_map, _ = pdata.get_gene_maps(on='protein') # type: ignore[attr-defined]

            resolved, unmatched = [], []
            var_names = pdata.prot.var_names.astype(str)

            for name in accessions:
                name = str(name)
                if name in var_names:
                    resolved.append(name)
                elif name in gene_map:
                    resolved.append(gene_map[name])
                else:
                    unmatched.append(name)

            if unmatched:
                warnings.warn(
                    f"The following accession(s) or gene name(s) were not found and will be ignored: {unmatched}"
                )

            if not resolved:
                warnings.warn("No matching accessions found. No proteins will be retained.")
                pdata.prot = pdata.prot[:, []]
                message_parts.append("accessions: 0 matched")
            else:
                pdata.prot = pdata.prot[:, pdata.prot.var_names.isin(resolved)]
                message_parts.append(f"accessions: {len(resolved)} matched / {len(accessions)} requested")

        # 3. Valid genes
        if valid_genes:
            # A. Remove invalid gene entries
            var = pdata.prot.var

            mask_missing_gene = var["Genes"].isna() | (var["Genes"].astype(str).str.strip() == "")
            keep_mask = ~mask_missing_gene

            if debug:
                print(f"Missing genes: {mask_missing_gene.sum()}")
                missing_names = pdata.prot.var_names[mask_missing_gene]
                print(f"Examples of proteins missing names: {missing_names[:5].tolist()}")

            pdata.prot = pdata.prot[:, keep_mask].copy()
            message_parts.append(f"valid_genes: removed {int(mask_missing_gene.sum())} proteins with invalid gene names")            

            # B. Resolve duplicate gene names
            var = pdata.prot.var  # refresh after filtering
            var_genes = var["Genes"].astype(str).str.strip()
            gene_counts = var_genes.value_counts()
            duplicates = gene_counts[gene_counts > 1].index.tolist()

            if len(duplicates) > 0:
                if debug:
                    print(f"Found {len(duplicates)} duplicate gene names.")

                # Track how many times each duplicate has appeared
                seen = {}
                new_names = []
                for gene in var["Genes"]:
                    if gene in duplicates:
                        seen[gene] = seen.get(gene, 0) + 1
                        if seen[gene] > 1:
                            gene = f"{gene}-{seen[gene]}"
                    new_names.append(gene)

                # Assign back to var
                pdata.prot.var["Genes"] = new_names

                message_parts.append(f"valid_genes: resolved {len(duplicates)} duplicate gene names by appending numeric suffixes")
                if debug:
                    example_dupes = [d for d in duplicates[:5]]
                    print(f"Examples of duplicate genes resolved: {example_dupes}")

        # 4. Remove duplicate profiles
        if unique_profiles:
            X = pdata.prot.X.toarray() if issparse(pdata.prot.X) else pdata.prot.X
            df_X = pd.DataFrame(X.T, index=pdata.prot.var_names)

            all_nan = np.all(np.isnan(X), axis=0)
            all_zero = np.all(X == 0, axis=0)
            empty_mask = all_nan | all_zero

            duplicated_mask = df_X.duplicated(keep="first").values  # mark duplicates
            
            # Combine removal conditions
            remove_mask = duplicated_mask | empty_mask
            keep_mask = ~remove_mask

            # Counts for each type
            n_dup = int(duplicated_mask.sum())
            n_empty = int(empty_mask.sum())
            n_total = int(remove_mask.sum())

            if debug:
                dup_names = pdata.prot.var_names[duplicated_mask]
                print(f"Duplicate abundance profiles detected: {n_dup} proteins")
                if len(dup_names) > 0:
                    print(f"Examples of duplicates: {dup_names[:5].tolist()}")
                print(f"Empty (all-zero or all-NaN) proteins detected: {n_empty}")

            # Apply filter
            pdata.prot = pdata.prot[:, keep_mask].copy()

            # Add summary message
            message_parts.append(
                f"unique_profiles: removed {n_dup} duplicate and {n_empty} empty abundance profiles "
                f"({n_total} total)"
            )

        if not message_parts:
            # no filters were applied
            message = f"{format_log_prefix('user')} Filtering proteins [failed]: {action} protein data.\n    → No filters applied."
        else:
            # at least 1 filter applied
            # PEPTIDES: also filter out peptides that belonged only to the filtered proteins
            if pdata.pep is not None and pdata.rs is not None: # type: ignore[attr-defined]
                proteins_to_keep, peptides_to_keep, orig_prot_names, orig_pep_names = pdata._filter_sync_peptides_to_proteins(
                    original=self, 
                    updated_prot=pdata.prot, 
                    debug=debug)

                # Apply filtered RS and update .prot and .pep using the helper
                pdata._apply_rs_filter(
                    keep_proteins=proteins_to_keep,
                    keep_peptides=peptides_to_keep,
                    orig_prot_names=orig_prot_names,
                    orig_pep_names=orig_pep_names,
                    debug=debug
                )

            # detect which filters were applied
            active_filters = []
            if condition is not None:
                active_filters.append("condition")
            if accessions is not None:
                active_filters.append("accession")
            if valid_genes:
                active_filters.append("valid genes")
            if unique_profiles:
                active_filters.append("unique profiles")

            # build the header, joining multiple filters nicely
            joined_filters = ", ".join(active_filters) if active_filters else "unspecified"
            message = (
                f"{format_log_prefix('user')} Filtering proteins [{joined_filters}]:\n"
                f"    {action} protein data based on {joined_filters}:"
            )

            for part in message_parts:
                    message += f"\n    → {part}"

            # Protein and peptide counts summary
            message += f"\n    → Proteins kept: {pdata.prot.shape[1]}"
            if pdata.pep is not None:
                message += f"\n    → Peptides kept (linked): {pdata.pep.shape[1]}\n"

        print(message)
        pdata._append_history(message) # type: ignore[attr-defined]
        pdata.update_summary(recompute=True) # type: ignore[attr-defined]
        return pdata if return_copy else None

    def filter_prot_found(self, group, min_ratio=None, min_count=None, on='protein', return_copy=True, verbose=True, match_any=False):
        """
        Filter proteins or peptides based on 'Found In' detection across samples or groups.

        This method filters features by checking whether they are found in a minimum number or proportion 
        of samples, either at the group level (e.g., biological condition) or based on individual files.

        Args:
            group (str or list of str): Group name(s) corresponding to 'Found In: {group} ratio' 
                (e.g., "HCT116_DMSO") or a list of filenames (e.g., ["F1", "F2"]). If this argument matches one or more `.obs` columns, the function automatically 
                interprets it as a class name, expands it to all class values, and annotates the
                necessary `'Found In:'` features.
            min_ratio (float, optional): Minimum proportion (0.0–1.0) of samples the feature must be 
                found in. Ignored for file-based filtering.
            min_count (int, optional): Minimum number of samples the feature must be found in. Alternative 
                to `min_ratio`. Ignored for file-based filtering.
            on (str): Feature level to filter: either "protein" or "peptide".
            return_copy (bool): If True, returns a filtered copy. If False, modifies in place.
            verbose (bool): If True, prints verbose summary information.
            match_any (bool): Defaults to False, for a AND search condition. If True, matches features found in any of the specified groups/files (i.e. union).

        Returns:
            pAnnData: A filtered pAnnData object if `return_copy=True`; otherwise, modifies in place and returns None.

        Note:
            - If `group` matches `.obs` column names, the method automatically annotates found 
              features by class before filtering.
            - For file-based filtering, use the file identifiers from `.prot.obs_names`.            

        Examples:
            Filter proteins found in both "groupA" and "groupB" groups, in at least 2 samples each:
                ```python
                pdata.filter_prot_found(group=["groupA", "groupB"], min_count=2)
                ```

            Filter proteins found in all three input files:
                ```python
                pdata.filter_prot_found(group=["F1", "F2", "F3"])
                ```

            Filter proteins found in files of a specific sub-group:
                ```python
                pdata.annotate_found(classes=['group','treatment'])
                pdata.filter_prot_found(group=["groupA_control", "groupB_treated"])
                ```

            If a single class column (e.g., `"cellline"`) is given, filter proteins based on each of its unique values (e.g. Line A, Line B):
                ```python
                pdata.filter_prot_found(group="cellline", min_ratio=0.5)
                ```
        """
        if not self._check_data(on): # type: ignore[attr-defined]
            return

        adata = self.prot if on == 'protein' else self.pep
        var = adata.var

        # Normalize group to list
        if isinstance(group, str):
            group = [group]
        if not isinstance(group, (list, tuple)):
            raise TypeError("`group` must be a string or list of strings.")

        # Auto-resolve obs columns passed instead of group values
        auto_value_msg = None
        if all(g in adata.obs.columns for g in group):
            if len(group) == 1:
                obs_col = group[0]
                expanded_groups = adata.obs[obs_col].unique().tolist()
            else:
                expanded_groups = (
                    adata.obs[group].astype(str)
                        .agg("_".join, axis=1)
                        .unique()
                        .tolist()
                )
            # auto-annotate found features by these obs columns
            self.annotate_found(classes=group, on=on, verbose=False)
            group = expanded_groups
            auto_value_msg = (
                f"{format_log_prefix('info', 2)} Found matching obs column(s): {group}. "
                "Automatically annotating detection by group values."
            )

        if verbose and auto_value_msg:
            print(auto_value_msg)

        # Determine filtering mode: group vs file or handle ambiguity/missing
        group_metrics = adata.uns.get(f"found_metrics_{on}")

        mode = None
        all_file_cols = all(f"Found In: {g}" in var.columns for g in group)
        all_group_cols = (
            group_metrics is not None
            and all((g, "count") in group_metrics.columns for g in group)
        )

        # --- 1️⃣ Explicit ambiguity: both file- and group-level indicators exist ---
        is_ambiguous, annotated_files, annotated_groups = _detect_ambiguous_input(group, var, group_metrics)
        if is_ambiguous:
            raise ValueError(
                f"Ambiguous input: items in {group} include both file identifiers {annotated_files} "
                f"and group values {annotated_groups}.\n"
                "Please separate group-based and file-based filters into separate calls."
            )

        # --- 2️⃣ Group-based mode ---
        elif all_group_cols:
            mode = "group"

        # --- 3️⃣ File-based mode ---
        elif all_file_cols:
            mode = "file"

        # --- 4️⃣ Mixed or unresolved case (fallback) ---
        else:
            missing = []
            for g in group:
                group_missing = (
                    group_metrics is None
                    or (g, "count") not in group_metrics.columns
                    or (g, "ratio") not in group_metrics.columns
                )
                file_missing = f"Found In: {g}" not in var.columns

                if group_missing and file_missing:
                    missing.append(g)

            # Consistent, readable user message
            msg = [f"The following group(s)/file(s) could not be found: {missing or '—'}"]
            msg.append("→ If these are group names, make sure you ran:")
            msg.append(f"   pdata.annotate_found(classes={group})")
            msg.append("→ If these are file names, ensure 'Found In: <file>' columns exist.\n")
            raise ValueError("\n".join(msg))

        # ---------------
        # Apply filtering
        mask = np.ones(len(var), dtype=bool)

        if mode == "file":
            if match_any: # OR logic
                mask = np.zeros(len(var), dtype=bool)
                for g in group:
                    col = f"Found In: {g}"
                    mask |= var[col]  
                if verbose:
                    print(f"{format_log_prefix('user')} Filtering proteins [Found|File-mode|ANY]: keeping {mask.sum()} / {len(mask)} features found in ANY of files: {group}")
            else: # AND logic (default)
                for g in group:
                    col = f"Found In: {g}"
                    mask &= var[col]
                if verbose:
                    print(f"{format_log_prefix('user')} Filtering proteins [Found|File-mode|ALL]: keeping {mask.sum()} / {len(mask)} features found in ALL files: {group}")

        elif mode == "group":
            if min_ratio is None and min_count is None:
                raise ValueError("You must specify either `min_ratio` or `min_count` when filtering by group.")
            
            if match_any: # ANY logic
                mask = np.zeros(len(var), dtype=bool)
                for g in group:
                    count_series = group_metrics[(g, "count")]
                    ratio_series = group_metrics[(g, "ratio")]

                    if min_ratio is not None:
                        this_mask = ratio_series >= min_ratio
                    else:
                        this_mask = count_series >= min_count

                    mask |= this_mask
                if verbose:
                    print(f"{format_log_prefix('user')} Filtering proteins [Found|Group-mode|ANY]: keeping {mask.sum()} / {len(mask)} features passing threshold in ANY of groups: {group}")

            else:
                for g in group:
                    count_series = group_metrics[(g, "count")]
                    ratio_series = group_metrics[(g, "ratio")]

                    if min_ratio is not None:
                        this_mask = ratio_series >= min_ratio
                    else:
                        this_mask = count_series >= min_count

                    mask &= this_mask

                if verbose:
                    print(f"{format_log_prefix('user')} Filtering proteins [Found|Group-mode|ALL]: keeping {mask.sum()} / {len(mask)} features passing threshold {min_ratio if min_ratio is not None else min_count} across groups: {group}")

        # Apply filtering
        filtered = self.copy() if return_copy else self # type: ignore[attr-defined], EditingMixin
        adata_filtered = adata[:, mask.values]

        if on == 'protein':
            filtered.prot = adata_filtered

            # Optional: filter peptides + rs as well
            if filtered.pep is not None and filtered.rs is not None:
                proteins_to_keep, peptides_to_keep, orig_prot_names, orig_pep_names = filtered._filter_sync_peptides_to_proteins(
                    original=self,
                    updated_prot=filtered.prot,
                    debug=verbose
                )

                filtered._apply_rs_filter(
                    keep_proteins=proteins_to_keep,
                    keep_peptides=peptides_to_keep,
                    orig_prot_names=orig_prot_names,
                    orig_pep_names=orig_pep_names,
                    debug=verbose
                )

        else:
            filtered.pep = adata_filtered
            # Optionally, we could also remove proteins no longer linked to any peptides,
            # but that's less common and we can leave it out unless requested.

        criteria_str = (
            f"min_ratio={min_ratio}" if mode == "group" and min_ratio is not None else
            f"min_count={min_count}" if mode == "group" else
            ("ANY files" if match_any else "ALL files")
        )

        logic_str = "ANY" if match_any else "ALL"

        filtered._append_history(  # type: ignore[attr-defined], HistoryMixin
            f"{on}: Filtered by detection in {mode} group(s) {group} using {criteria_str} (match_{logic_str})."
        )
        filtered.update_summary(recompute=True) # type: ignore[attr-defined], SummaryMixin

        return filtered if return_copy else None

    def filter_prot_significant(self, group=None, min_ratio=None, min_count=None, fdr_threshold=0.01, return_copy=True, verbose=True, match_any=True):
        """
        Filter proteins based on significance across samples or groups using FDR thresholds.

        This method filters proteins by checking whether they are significant (e.g. PG.Q.Value < 0.01)
        in a minimum number or proportion of samples, either per file or grouped.

        Args:
            group (str, list, or None): Group name(s) (e.g., sample classes or filenames). If None, uses all files.
            min_ratio (float, optional): Minimum proportion of samples to be significant.
            min_count (int, optional): Minimum number of samples to be significant.
            fdr_threshold (float): Significance threshold (default = 0.01).
            return_copy (bool): Whether to return a filtered copy or modify in-place.
            verbose (bool): Whether to print summary.
            match_any (bool): If True, retain proteins significant in *any* group/file (OR logic). If False, require *all* groups/files to be significant (AND logic).
            
        Returns:
            pAnnData or None: Filtered object (if `return_copy=True`) or modifies in-place.

        Examples:
            Filter proteins significant in both "groupA" and "groupB" groups, FDR of 0.01 (default):
                ```python
                pdata.filter_prot_significant(group=["groupA", "groupB"], min_count=2)
                ```

            Filter proteins significant in all three input files:
                ```
                pdata.filter_prot_significant(group=["F1", "F2", "F3"])
                ```

            Filter proteins significant in files of a specific sub-group:
                ```python
                pdata.annotate_significant(classes=['group','treatment'])
                pdata.filter_prot_significant(group=["groupA_control", "groupB_treated"])            
                ```
                    
        Todo:
            Implement peptide then protein filter
        """
        if not self._check_data("protein"): # type: ignore[attr-defined]
            return

        adata = self.prot 
        var = adata.var

        # Detect per-sample significance layer
        has_protein_level_significance = any(
            k.lower().endswith("_qval") or k.lower().endswith("_fdr") for k in adata.layers.keys()
        )

        # --- Handle missing significance data entirely ---
        if not has_protein_level_significance and "Global_Q_value" not in adata.var.columns:
            raise ValueError(
                "No per-sample layer (e.g., *_qval) or global significance column ('Global_Q_value') "
                "found in .prot. Please ensure your data includes q-values or run annotate_significant()."
            )

        # --- 1️⃣ Global fallback mode (e.g. PD-based imports) ---
        if not has_protein_level_significance and "Global_Q_value" in adata.var.columns:
            if group is not None:
                raise ValueError(
                    f"Cannot filter by group {group}: per-sample significance data missing "
                    "and only global q-values available."
                )
            
            global_mask = adata.var["Global_Q_value"] < fdr_threshold

            n_total = len(global_mask)
            n_kept = int(global_mask.sum())
            n_dropped = n_total - n_kept

            filtered = self.copy() if return_copy else self
            filtered.prot = adata[:, global_mask]

            if filtered.pep is not None and filtered.rs is not None:
                proteins_to_keep, peptides_to_keep, orig_prot_names, orig_pep_names = filtered._filter_sync_peptides_to_proteins(
                    original=self, updated_prot=filtered.prot, debug=verbose
                )
                filtered._apply_rs_filter(
                    keep_proteins=proteins_to_keep,
                    keep_peptides=peptides_to_keep,
                    orig_prot_names=orig_prot_names,
                    orig_pep_names=orig_pep_names,
                    debug=verbose,
                )

            filtered.update_summary(recompute=True)
            filtered._append_history(
                f"Filtered by global significance (Global_Q_value < {fdr_threshold}); "
                f"{n_kept}/{n_total} proteins retained."
            )

            if verbose:
                print(f"{format_log_prefix('user')} Filtering proteins by significance [Global-mode]:")
                print(f"{format_log_prefix('info', 2)} Using global protein-level q-values (no per-sample significance available).")
                return_copy_str = "Returning a copy of" if return_copy else "Filtered and modified"
                print(f"    {return_copy_str} protein data based on significance thresholds:")
                print(f"{format_log_prefix('filter_conditions')}Files requested: All")
                print(f"{format_log_prefix('filter_conditions')}FDR threshold: {fdr_threshold}")
                print(f"    → Proteins kept: {n_kept}, Proteins dropped: {n_dropped}\n")

            return filtered if return_copy else None

        # --- 2️⃣ Per-sample significance data available ---
        no_group_msg = None
        auto_group_msg = None
        auto_value_msg = None

        if group is None:
            group_list = list(adata.obs_names)
            if verbose:
                no_group_msg = f"{format_log_prefix('info', 2)} No group provided. Defaulting to sample-level significance filtering."
        else:
            group_list = [group] if isinstance(group, str) else group
            
        # Ensure annotations exist or auto-generate
        missing_cols = [f"Significant In: {g}" for g in group_list]
        if all(col in var.columns for col in missing_cols):
            # Case A: user passed actual group values, already annotated
            pass
        else:
            # Case B: need to resolve automatically
            if all(g in adata.obs.columns for g in group_list):
                # User passed obs column(s)
                if len(group_list) == 1:
                    obs_col = group_list[0]
                    expanded_groups = adata.obs[obs_col].unique().tolist()
                else:
                    expanded_groups = (
                        adata.obs[group_list].astype(str)
                            .agg("_".join, axis=1)
                            .unique()
                            .tolist()
                    )
                self.annotate_significant(classes=group_list,
                                        fdr_threshold=fdr_threshold,
                                        on="protein", verbose=False)
                group_list = expanded_groups
                auto_group_msg = f"{format_log_prefix('info', 2)} Found matching obs column '{group_list}'. Automatically annotating significance by group: {group_list} using FDR threshold {fdr_threshold}."

            else:
                # User passed group values, but not annotated yet
                found_obs_col = None
                for obs_col in adata.obs.columns:
                    if set(group_list).issubset(set(adata.obs[obs_col].unique())):
                        found_obs_col = obs_col
                        break

                if found_obs_col is not None:
                    self.annotate_significant(classes=[found_obs_col],
                                            fdr_threshold=fdr_threshold,
                                            on="protein", indent=2, verbose=False)
                    auto_value_msg = (f"{format_log_prefix('info', 2)} Found matching obs column '{found_obs_col}'"
                    f"for groups {group_list}. Automatically annotating significant features by group {found_obs_col} "
                    f"using FDR threshold {fdr_threshold}.")    
                else:
                    raise ValueError(
                        f"Could not find existing significance annotations for groups {group_list}. "
                        "Please either pass valid obs column(s), provide values from a valid `.obs` column or run `annotate_significant()` first."
                    )

        # --- 3️⃣ Mode detection and ambiguity handling ---
        metrics_key = "significance_metrics_protein"
        metrics_df = adata.uns.get(metrics_key, pd.DataFrame())

        is_ambiguous, annotated_files, annotated_groups = _detect_ambiguous_input(group_list, var, metrics_df)
        if is_ambiguous:
            raise ValueError(
                f"Ambiguous input: items in {group_list} include both file identifiers {annotated_files} "
                f"and group values {annotated_groups}.\n"
                "Please separate group-based and file-based filters into separate calls."
            )

        all_group_cols = (
            metrics_df is not None
            and all((g, "count") in metrics_df.columns for g in group_list)
        )
        all_file_cols = all(f"Significant In: {g}" in var.columns for g in group_list)
        mode = "group" if all_group_cols else "file"

        # Build filtering mask
        mask = np.zeros(len(var), dtype=bool) if match_any else np.ones(len(var), dtype=bool)

        if mode == "group":
            if min_ratio is None and min_count is None:
                raise ValueError("Specify `min_ratio` or `min_count` for group-based filtering.")
            for g in group_list:
                count = metrics_df[(g, "count")]
                ratio = metrics_df[(g, "ratio")]
                this_mask = ratio >= min_ratio if min_ratio is not None else count >= min_count
                mask = mask | this_mask if match_any else mask & this_mask
        else:  # file mode
            for g in group_list:
                col = f"Significant In: {g}"
                this_mask = var[col].values
                mask = mask | this_mask if match_any else mask & this_mask

        # --- 4️⃣ Apply filtering and sync ---
        filtered = self.copy() if return_copy else self
        filtered.prot = adata[:, mask]

        # Sync peptides and RS
        if filtered.pep is not None and filtered.rs is not None:
            proteins_to_keep, peptides_to_keep, orig_prot_names, orig_pep_names = filtered._filter_sync_peptides_to_proteins(
                original=self, updated_prot=filtered.prot, debug=verbose
            )
            filtered._apply_rs_filter(
                keep_proteins=proteins_to_keep,
                keep_peptides=peptides_to_keep,
                orig_prot_names=orig_prot_names,
                orig_pep_names=orig_pep_names,
                debug=verbose
            )

        filtered.update_summary(recompute=True)
        filtered._append_history(
            f"Filtered by significance (FDR < {fdr_threshold}) in group(s): {group_list}, "
            f"using min_ratio={min_ratio} / min_count={min_count}, match_any={match_any}"
        )

        if verbose:
            logic = "any" if match_any else "all"
            mode_str = "Group-mode" if mode == "group" else "File-mode"

            print(f"{format_log_prefix('user')} Filtering proteins [Significance|{mode_str}]:")

            if no_group_msg:
                print(no_group_msg)
            if auto_group_msg:
                print(auto_group_msg)
            if auto_value_msg:
                print(auto_value_msg)

            return_copy_str = "Returning a copy of" if return_copy else "Filtered and modified"
            print(f"    {return_copy_str} protein data based on significance thresholds:")

            if mode == "group":
                # Case A: obs column(s) expanded → show expanded_groups and add note
                if auto_group_msg:
                    group_note = f" (all values of obs column(s))"
                    print(f"{format_log_prefix('filter_conditions')}Groups requested: {group_list}{group_note}")
                else:
                    print(f"{format_log_prefix('filter_conditions')}Groups requested: {group_list}")
                print(f"{format_log_prefix('filter_conditions')}FDR threshold: {fdr_threshold}")
                if min_ratio is not None:
                    print(f"{format_log_prefix('filter_conditions')}Minimum ratio: {min_ratio} (match_{logic} = {match_any})")
                if min_count is not None:
                    print(f"{format_log_prefix('filter_conditions')}Minimum count: {min_count} (match_{logic} = {match_any})")
            else:
                print(f"{format_log_prefix('filter_conditions')}Files requested: All")
                print(f"{format_log_prefix('filter_conditions')}FDR threshold: {fdr_threshold}")
                print(f"{format_log_prefix('filter_conditions')}Logic: {logic} "
                    f"(protein must be significant in {'≥1' if match_any else 'all'} file(s))")

            n_kept = int(mask.sum())
            n_total = len(mask)
            n_dropped = n_total - n_kept
            print(f"    → Proteins kept: {n_kept}, Proteins dropped: {n_dropped}\n")

        return filtered if return_copy else None

    def _filter_sync_peptides_to_proteins(self, original, updated_prot, debug=None):
        """
        Helper function to filter peptides based on the updated protein list.

        This method determines which peptides to retain after protein-level filtering,
        and returns the necessary inputs for `_apply_rs_filter`.

        Args:
            original (pAnnData): Original pAnnData object before filtering.
            updated_prot (AnnData): Updated protein AnnData object to filter against.
            debug (bool, optional): If True, prints debugging information.

        Returns:
            tuple: Inputs needed for downstream `_apply_rs_filter` operation.
        """
        if debug:
            print(f"{format_log_prefix('info')} Applying RS-based peptide sync-up on peptides after protein filtering...")

        # Get original axis names from unfiltered self
        rs = original.rs
        orig_prot_names = np.array(original.prot.var_names)
        orig_pep_names = np.array(original.pep.var_names)
        # Determine which protein rows to keep in RS
        proteins_to_keep=updated_prot.var_names
        keep_set = set(proteins_to_keep)
        prot_mask = np.fromiter((p in keep_set for p in orig_prot_names), dtype=bool)
        rs_filtered = rs[prot_mask, :]
        # Keep peptides that are still linked to ≥1 protein
        pep_mask = np.array(rs_filtered.sum(axis=0)).ravel() > 0
        peptides_to_keep = orig_pep_names[pep_mask]

        return proteins_to_keep, peptides_to_keep, orig_prot_names, orig_pep_names

    def filter_sample(self, values=None, exact_cases=False, condition=None, file_list=None, exclude_file_list=None, min_prot=None, cleanup=True, return_copy=True, debug=False, query_mode=False):
        """
        Filter samples in a pAnnData object based on categorical, numeric, or identifier-based criteria.

        You must specify **exactly one** of the following:
        
        - `values`: Dictionary or list of dictionaries specifying class-based filters (e.g., treatment, cellline).
        - `condition`: A string condition evaluated against summary-level numeric metadata (e.g., protein count).
        - `file_list`: List of sample or file names to retain.

        Args:
            values (dict or list of dict, optional): Categorical metadata filter. Matches rows in `.summary` or `.obs` with those field values.
                Examples: `{'treatment': 'kd', 'cellline': 'A'}`.
            exact_cases (bool): If True, uses exact match across all class values when `values` is a list of dicts.
            condition (str, optional): Logical condition string referencing summary columns. This should reference columns in `pdata.summary`.
                Examples: `"protein_count > 1000"`.
            file_list (list of str, optional): List of sample names or file identifiers to keep. Filters to only those samples (must match obs_names).
            exclude_file_list (list of str, optional): Similar to `file_list`, but excludes the specified files/samples instead of keeping them.
            min_prot (int, optional): Minimum number of proteins required in a sample to retain it.
            cleanup (bool): If True (default), remove proteins that become all-NaN or all-zero after sample filtering and synchronize RS/peptide matrices. Set to False to retain all proteins for consistent feature alignment (e.g. during DE analysis).
            return_copy (bool): If True, returns a filtered pAnnData object; otherwise modifies in place.
            debug (bool): If True, prints query strings and filter summaries.
            query_mode (bool): If True, interprets `values` or `condition` as a raw pandas-style `.query()` string and evaluates it directly on `.obs` or `.summary` respectively.

        Returns:
            pAnnData: Filtered pAnnData object if `return_copy=True`; otherwise, modifies in place and returns None.

        Raises:
            ValueError: If more than one or none of `values`, `condition`, or `file_list` is specified.

        Examples:
            Filter by metadata values:
                ```python
                pdata.filter_sample(values={'treatment': 'kd', 'cellline': 'A'})
                ```

            Filter with multiple exact matching cases:
                ```python
                pdata.filter_sample(
                    values=[
                        {'treatment': 'kd', 'cellline': 'A'},
                        {'treatment': 'sc', 'cellline': 'B'}
                    ],
                    exact_cases=True
                )
                ```

            Filter by numeric condition on summary:
                ```python
                pdata.filter_sample(condition="protein_count > 1000")
                ```

            Filter samples with fewer than 1000 proteins:
                ```python
                pdata.filter_sample(min_prot=1000)
                ```

            Keep specific samples by name:
                ```python
                pdata.filter_sample(file_list=['Sample_001', 'Sample_007'])
                ```

            Exclude specific files from the dataset:
                ```python
                pdata.filter_sample(exclude_file_list=['Sample_001', 'Sample_007'])
                ```

            For advanced usage using query mode, see the note below.
            
            !!! note "Advanced Usage"
                To enable **advanced filtering**, set `query_mode=True` to evaluate raw pandas-style queries:

                - Query `.obs` metadata:
                    ```python
                    pdata.filter_sample(values="cellline == 'AS' and treatment == 'kd'", query_mode=True)
                    ```

                - Query `.summary` metadata:
                    ```python
                    pdata.filter_sample(condition="protein_count > 1000 and missing_pct < 0.2", query_mode=True)
                    ```            
        """
        # Ensure exactly one of the filter modes is specified
        provided = [values, condition, file_list, min_prot, exclude_file_list]
        if sum(arg is not None for arg in provided) != 1:
            raise ValueError(
                "Invalid filter input. You must specify exactly one of the following keyword arguments:\n"
                "- `values=...` for categorical metadata filtering,\n"
                "- `condition=...` for summary-level condition filtering, or\n"
                "- `min_prot=...` to filter by minimum protein count.\n"
                "- `file_list=...` to filter by sample IDs.\n"
                "- `exclude_file_list=...` to exclude specific sample IDs.\n\n"
                "Examples:\n"
                "  pdata.filter_sample(condition='protein_quant > 0.2')"
            )

        if min_prot is not None:
            condition = f"protein_count >= {min_prot}"

        if values is not None and not query_mode:
            return self._filter_sample_values(
                values=values,
                exact_cases=exact_cases,
                debug=debug,
                return_copy=return_copy, 
                cleanup=cleanup
            )

        if (condition is not None or file_list is not None or exclude_file_list is not None) and not query_mode:
            return self._filter_sample_condition(
                condition=condition,
                file_list=file_list,
                exclude_file_list=exclude_file_list,
                return_copy=return_copy,
                debug=debug, 
                cleanup=cleanup
            )
        
        if values is not None and query_mode:
            return self._filter_sample_query(query_string=values, source='obs', return_copy=return_copy, debug=debug, cleanup=cleanup)

        if condition is not None and query_mode:
            return self._filter_sample_query(query_string=condition, source='summary', return_copy=return_copy, debug=debug, cleanup=cleanup)

    def _filter_sample_condition(self, condition = None, return_copy = True, file_list=None, exclude_file_list=None, cleanup=True, debug=False):
        """
        Filter samples based on numeric metadata conditions or a list of sample identifiers.

        This internal method supports two modes:
        
        - A string `condition` evaluated against `.summary` (e.g., `"protein_count > 1000"`).
        - A `file_list` of sample names or identifiers to retain (e.g., filenames or `.obs_names`).

        Args:
            condition (str, optional): Logical condition string referencing columns in `.summary`.
            file_list (list of str, optional): List of sample identifiers to keep.
            exclude_file_list (list of str, optional): Same as file_list, but excludes specified sample identifiers.
            cleanup (bool): If True (default), remove proteins that become all-NaN or all-zero
                after sample filtering and synchronize RS/peptide matrices. Set to False to
                retain all proteins for consistent feature alignment (e.g. during DE analysis).
            return_copy (bool): If True, returns a filtered pAnnData object. If False, modifies in place.
            debug (bool): If True, prints the query string or filtering summary.

        Returns:
            pAnnData: Filtered pAnnData object if `return_copy=True`; otherwise, modifies in place and returns None.

        Note:
            This method is intended for internal use by `filter_sample()`. For general-purpose filtering, 
            use `filter_sample()` with `condition=...` or `file_list=...`.

        Examples:
            Filter samples with more than 1000 proteins:
                ```python
                pdata.filter_sample_condition(condition="protein_count > 1000")
                ```

            Keep only specific sample files:
                ```python
                pdata.filter_sample_condition(file_list=['fileA', 'fileB'])
                ```

            Exclude specific files from the dataset:
                ```python
                pdata.filter_sample(exclude_file_list=['Sample_001', 'Sample_007'])
                ```
        """
        if not self._has_data(): # type: ignore[attr-defined], ValidationMixin
            pass

        if self._summary is None: # type: ignore[attr-defined]
            self.update_summary(recompute=True) # type: ignore[attr-defined], SummaryMixin
        
        if file_list is not None and exclude_file_list is not None:
            raise ValueError(
                "You cannot specify both `file_list` and `exclude_file_list` simultaneously.\n"
                "Please use only one mode per call:\n"
                "  • `file_list=[...]` to keep only these samples, or\n"
                "  • `exclude_file_list=[...]` to remove these samples.\n"
                "If you need both operations, call `filter_sample()` twice in sequence."
            )

        # Determine whether to operate on a copy or in-place
        pdata = self.copy() if return_copy else self # type: ignore[attr-defined], EditingMixin
        action = "Returning a copy of" if return_copy else "Filtered and modified"

        orig_sample_count = len(pdata.prot.obs)

        if debug:
            print("self.prot id:", id(self.prot))
            print("pdata.prot id:", id(pdata.prot))
            print("Length of pdata.prot.obs_names:", len(pdata.prot.obs_names))

        # Determine sample indices to retain
        index_filter = None
        missing = []

        if condition is not None:
            formatted_condition = self._format_filter_query(condition, pdata._summary)  # type: ignore[attr-defined]
            if debug:
                print(formatted_condition)
            index_filter = pdata._summary[pdata._summary.eval(formatted_condition)].index
        elif file_list is not None or exclude_file_list is not None:
            all_samples = set(pdata.prot.obs_names)

            if file_list is not None:
                requested = set(file_list)
                missing = list(requested - all_samples)
                index_filter = list(all_samples.intersection(requested))
                mode_str = "including"
                exclude_mode = False

            else:  # exclude_file_list is not None
                requested = set(exclude_file_list)
                missing = list(requested - all_samples)
                index_filter = list(all_samples - requested)
                mode_str = "excluding"
                exclude_mode = True

            if missing:
                warnings.warn(f"Some sample IDs not found: {missing}")

        else:
            # No filtering applied
            message = "No filtering applied. Returning original data."
            return pdata if return_copy else None

        if debug:
            print(f"Length of index_filter: {len(index_filter)}")
            print(f"Length of pdata.prot.obs_names before filter: {len(pdata.prot.obs_names)}")
            print(f"Number of shared samples: {len(pdata.prot.obs_names.intersection(index_filter))}")

        # Filter out selected samples from prot and pep
        if pdata.prot is not None:
            pdata.prot = pdata.prot[pdata.prot.obs.index.isin(index_filter)]
        
        if pdata.pep is not None:
            pdata.pep = pdata.pep[pdata.pep.obs.index.isin(index_filter)]

        if cleanup:
            cleanup_message = pdata._cleanup_proteins_after_sample_filter(verbose=True)
        else:
            cleanup_message = None
        pdata.update_summary(recompute=False, verbose=False) # type: ignore[attr-defined], SummaryMixin

        print(f"Length of pdata.prot.obs_names after filter: {len(pdata.prot.obs_names)}") if debug else None

        # Construct formatted message
        filter_type = "condition" if condition else "file list" if (file_list or exclude_file_list) else "none"
        log_prefix = format_log_prefix("user")

        if len(index_filter) == 0:
            message = f"{log_prefix} Filtering samples [{filter_type}]:\n    → No matching samples found. No filtering applied."
        else:
            message = f"{log_prefix} Filtering samples [{filter_type}]:\n"
            message += f"    {action} sample data based on {filter_type}:\n"
            if condition:
                message += f"{format_log_prefix('filter_conditions')}Condition: {condition}\n"
            elif file_list or exclude_file_list:
                flist = file_list if file_list is not None else exclude_file_list
                message += f"{format_log_prefix('filter_conditions')}Files requested ({mode_str}): {len(flist)}\n"
                if missing:
                    message += f"{format_log_prefix('filter_conditions')}Missing samples ignored: {len(missing)}\n"

            message += cleanup_message + "\n" if cleanup_message else ""
            message += f"    → Samples kept: {len(pdata.prot.obs)}, Samples dropped: {orig_sample_count - len(pdata.prot.obs)}"
            message += f"\n    → Proteins kept: {len(pdata.prot.var)}\n"

        # Logging and history updates
        print(message)
        pdata._append_history(message) # type: ignore[attr-defined], HistoryMixin

        return pdata if return_copy else None

    def _filter_sample_values(self, values, exact_cases, cleanup=True, verbose=True, debug=False, return_copy=True):
        """
        Filter samples using dictionary-style categorical matching.

        This internal method filters samples based on class-like annotations (e.g., treatment, cellline),
        using either loose field-wise filtering or strict combination matching. It supports:
    
        - Single dictionary (e.g., `{'cellline': 'A'}`)
        - List of dictionaries (e.g., `[{...}, {...}]` for multiple matching cases)
        - Exact matching (`exact_cases=True`) across all key–value pairs

        Args:
            values (dict or list of dict): Filtering conditions.
                - If `exact_cases=False`: A single dictionary with field: list of values. 
                Applies OR logic within fields and AND logic across fields.
                - If `exact_cases=True`: A list of dictionaries, each representing an exact combination of field values.
            exact_cases (bool): If True, performs exact match filtering using the provided list of dictionaries.
            cleanup (bool): If True (default), remove proteins that become all-NaN or all-zero
                after sample filtering and synchronize RS/peptide matrices. Set to False to
                retain all proteins for consistent feature alignment (e.g. during DE analysis).
            verbose (bool): If True, prints a summary of the filtering result.
            debug (bool): If True, prints internal queries and matching logic.
            return_copy (bool): If True, returns a filtered copy. Otherwise modifies in place.

        Returns:
            pAnnData: Filtered view of the input AnnData object if `return_copy=True`; otherwise modifies in place and returns None.

        Note:
            This method is used internally by `filter_sample()`. For general use, call `filter_sample()` directly.

        Examples:
            Loose field-wise match (OR within fields, AND across fields):
                ```python
                pdata.filter_sample_values(values={'treatment': ['kd', 'sc'], 'cellline': 'A'})
                ```

            Exact combination matching:
                ```python
                pdata.filter_sample_values(
                    values=[
                        {'treatment': 'kd', 'cellline': 'A'},
                        {'treatment': 'sc', 'cellline': 'B'}
                    ],
                    exact_cases=True
                )
                ```
        """

        pdata = self.copy() if return_copy else self # type: ignore[attr-defined], EditingMixin
        obs_keys = pdata.summary.columns # type: ignore[attr-defined]
        orig_sample_count = len(pdata.prot.obs)

        if exact_cases:
            if not isinstance(values, list) or not all(isinstance(v, dict) for v in values):
                raise ValueError("When exact_cases=True, `values` must be a list of dictionaries.")

            for case in values:
                if not case:
                    raise ValueError("Empty dictionary found in values.")
                for key in case:
                    if key not in obs_keys:
                        raise ValueError(f"Field '{key}' not found in adata.obs.")

            query = " | ".join([
                " & ".join([
                    f"(adata.obs['{k}'] == '{v}')" for k, v in case.items()
                ])
                for case in values
            ])

        else:
            if not isinstance(values, dict):
                raise ValueError("When exact_cases=False, `values` must be a dictionary.")

            for key in values:
                if key not in obs_keys:
                    raise ValueError(f"Field '{key}' not found in adata.obs.")

            query_parts = []
            for k, v in values.items():
                v_list = v if isinstance(v, list) else [v]
                part = " | ".join([f"(adata.obs['{k}'] == '{val}')" for val in v_list])
                query_parts.append(f"({part})")
            query = " & ".join(query_parts)

        if debug:
                print(f"Filter query: {query}")

        if pdata.prot is not None:
            adata = pdata.prot
            pdata.prot = adata[eval(query)]
        if pdata.pep is not None:
            adata = pdata.pep
            pdata.pep = adata[eval(query)]
        
        if cleanup:
            cleanup_message = pdata._cleanup_proteins_after_sample_filter(verbose=True)
        else:
            cleanup_message = None
        pdata.update_summary(recompute=False, verbose=False) # type: ignore[attr-defined], SummaryMixin

        n_samples = len(pdata.prot)
        log_prefix = format_log_prefix("user")
        filter_mode = "exact match" if exact_cases else "class match"

        if n_samples == 0:
            message = (
                f"{log_prefix} Filtering samples [{filter_mode}]:\n"
                f"    → No matching samples found. No filtering applied."
            )
        else:
            message = (
                f"{log_prefix} Filtering samples [{filter_mode}]:\n"
                f"    {'Returning a copy of' if return_copy else 'Filtered and modified'} sample data based on {filter_mode}:\n"
            )

            if exact_cases:
                message += f"{format_log_prefix('filter_conditions')}Matching any of the following cases:\n"
                for i, case in enumerate(values, 1):
                    message += f"       {i}. {case}\n"
            else:
                message += "   🔸 Match samples where:\n"
                for k, v in values.items():
                    valstr = v if isinstance(v, str) else ", ".join(map(str, v))
                    message += f"      - {k}: {valstr}\n"

            message += cleanup_message + "\n" if cleanup_message else ""
            message += f"    → Samples kept: {n_samples}, Samples dropped: {orig_sample_count - n_samples}"
            message += f"\n    → Proteins kept: {len(pdata.prot.var)}\n"

        print(message) if verbose else None
        pdata._append_history(message) # type: ignore[attr-defined], HistoryMixin

        return pdata

    def _filter_sample_query(self, query_string, source='obs', cleanup=True, return_copy=True, debug=False):
        """
        Filter samples using a raw pandas-style query string on `.obs` or `.summary`.

        This method allows advanced filtering of samples using logical expressions evaluated 
        directly on the sample metadata.

        Args:
            query_string (str): A pandas-style query string. 
                Examples: `"cellline == 'AS' and treatment in ['kd', 'sc']"`.
            source (str): The metadata source to query — either `"obs"` or `"summary"`.
            cleanup (bool): If True (default), remove proteins that become all-NaN or all-zero
                after sample filtering and synchronize RS/peptide matrices. Set to False to
                retain all proteins for consistent feature alignment (e.g. during DE analysis).
            return_copy (bool): If True, returns a filtered pAnnData object; otherwise modifies in place.
            debug (bool): If True, prints the parsed query and debug messages.

        Returns:
            pAnnData: Filtered pAnnData object if `return_copy=True`; otherwise, modifies in place and returns None.
        """
        pdata = self.copy() if return_copy else self # type: ignore[attr-defined], EditingMixin
        action = "Returning a copy of" if return_copy else "Filtered and modified"
        orig_sample_count = len(pdata.prot.obs)

        print(f"{format_log_prefix('warn',indent=1)} Advanced query mode enabled — interpreting string as a pandas-style expression.")

        if source == 'obs':
            df = pdata.prot.obs
        elif source == 'summary':
            if self._summary is None: # type: ignore[attr-defined]
                self.update_summary(recompute=True) # type: ignore[attr-defined], SummaryMixin
            df = pdata._summary # type: ignore[attr-defined]
        else:
            raise ValueError("source must be 'obs' or 'summary'")

        try:
            filtered_df = df.query(query_string)
        except Exception as e:
            raise ValueError(f"Failed to parse query string:\n  {query_string}\nError: {e}")

        index_filter = filtered_df.index

        if pdata.prot is not None:
            pdata.prot = pdata.prot[pdata.prot.obs_names.isin(index_filter)]
        if pdata.pep is not None:
            pdata.pep = pdata.pep[pdata.pep.obs_names.isin(index_filter)]
        
        if cleanup:
            cleanup_message = pdata._cleanup_proteins_after_sample_filter(verbose=True)
        else:
            cleanup_message = None
        pdata.update_summary(recompute=False, verbose=False) # type: ignore[attr-defined], SummaryMixin

        n_samples = len(pdata.prot)
        log_prefix = format_log_prefix("user")
        action = "Returning a copy of" if return_copy else "Filtered and modified"

        message = (
            f"{log_prefix} Filtering samples [query]:\n"
            f"    {action} sample data based on query string:\n"
            f"   🔸 Query: {query_string}\n"
        )

        if cleanup_message:
            message += f"{cleanup_message}\n"
        
        message += (
            f"    → Samples kept: {n_samples}, Samples dropped: {orig_sample_count - n_samples}\n"
            f"    → Proteins kept: {len(pdata.prot.var)}\n"
        )

        print(message)

        history_message = f"{action} samples based on query string. Samples kept: {len(index_filter)}."
        pdata._append_history(history_message) # type: ignore[attr-defined], HistoryMixin

        return pdata if return_copy else None

    def _cleanup_proteins_after_sample_filter(self, verbose=True, printout=False):
        """
        Internal helper to remove proteins that became all-NaN or all-zero
        after sample filtering. Called silently by `filter_sample()`.

        Ensures the RS matrix and peptide table are synchronized after cleanup.
        Called automatically by `filter_sample()` and during import.        

        Args:
            verbose (bool): If True, returns a cleanup summary message.
                            If False, runs silently.
            printout (bool): If True and verbose=True, directly prints the cleanup mesage.

        Returns:
            str or None: Cleanup message if verbose=True and any proteins removed,
                        otherwise None.
        """
        from scipy.sparse import issparse
        
        if not self._check_data("protein"):  # type: ignore[attr-defined]
            return

        X = self.prot.X.toarray() if issparse(self.prot.X) else self.prot.X
        original_var_names = self.prot.var_names.copy()
        all_nan = np.all(np.isnan(X), axis=0)
        all_zero = np.all(X == 0, axis=0)
        remove_mask = all_nan | all_zero

        if not remove_mask.any():
            if verbose:
                return f"{format_log_prefix('info_only',2)} Auto-cleanup: No empty proteins found (all-NaN or all-zero)."
            else:
                return None

        n_remove = int(remove_mask.sum())
        keep_mask = ~remove_mask

        # skip cleanup entirely if no samples or no protein data remain
        if self.prot is None or self.prot.n_obs == 0 or self.prot.n_vars == 0:
            if verbose:
                print(f"{format_log_prefix('warn_only',2)} No samples or proteins to clean up. Skipping RS sync.")
            return None

        # Backup original for RS/peptide syncing, ensure summary and obs are aligned before making copy
        if self._summary is not None and not self.prot.obs.index.equals(self._summary.index):
            self._summary = self._summary.loc[self.prot.obs.index].copy()
        original = self.copy()
        # Filter protein data
        self.prot = self.prot[:, keep_mask].copy()

        if self.pep is not None and self.rs is not None:
            proteins_to_keep, peptides_to_keep, orig_prot_names, orig_pep_names = self._filter_sync_peptides_to_proteins(
                original=original,
                updated_prot=self.prot,
                debug=False
            )

            self._apply_rs_filter(
                keep_proteins=proteins_to_keep,
                keep_peptides=peptides_to_keep,
                orig_prot_names=orig_prot_names,
                orig_pep_names=orig_pep_names,
                debug=False
            )

        self.update_summary(recompute=True, verbose=False)

        if verbose:            
            removed_proteins = list(original_var_names[remove_mask])
            preview = ", ".join(removed_proteins[:10])
            if n_remove > 10:
                preview += ", ..."

            if printout and n_remove>0:
                # for startup
                print(f"{format_log_prefix('info_only',1)} Removed {n_remove} empty proteins (all-NaN or all-zero). Proteins: {preview}")
            else:
                # for filter
                return(f"{format_log_prefix('info_only',2)} Auto-cleanup: Removed {n_remove} empty proteins (all-NaN or all-zero). Proteins: {preview}")
        
        return None

    def filter_rs(
        self,
        min_peptides_per_protein=None,
        min_unique_peptides_per_protein=2,
        max_proteins_per_peptide=None,
        return_copy=True,
        preset=None,
        validate_after=True
    ):
        """
        Filter the RS matrix and associated `.prot` and `.pep` data based on peptide-protein relationships.

        This method applies rules for keeping proteins with sufficient peptide evidence and/or removing
        ambiguous peptides. It also updates internal mappings accordingly.

        Args:
            min_peptides_per_protein (int, optional): Minimum number of total peptides required per protein.
            min_unique_peptides_per_protein (int, optional): Minimum number of unique peptides required per protein 
                (default is 2).
            max_proteins_per_peptide (int, optional): Maximum number of proteins a peptide can map to; peptides 
                exceeding this will be removed.
            return_copy (bool): If True (default), returns a filtered pAnnData object. If False, modifies in place.
            preset (str or dict, optional): Predefined filter presets:
                - `"default"` → unique peptides ≥ 2
                - `"lenient"` → total peptides ≥ 2
                - A dictionary specifying filter thresholds manually.
            validate_after (bool): If True (default), calls `self.validate()` after filtering.

        Returns:
            pAnnData: Filtered pAnnData object if `return_copy=True`; otherwise, modifies in place and returns None.

        Note:
            Stores filter metadata in `.prot.uns['filter_rs']`, including indices of proteins/peptides kept and 
            filtering summary.
        """
        if self.rs is None: # type: ignore[attr-defined]
            print("⚠️ No RS matrix to filter.")
            return self if return_copy else None

        # --- Apply preset if given ---
        if preset:
            if preset == "default":
                min_peptides_per_protein = None
                min_unique_peptides_per_protein = 2
                max_proteins_per_peptide = None
            elif preset == "lenient":
                min_peptides_per_protein = 2
                min_unique_peptides_per_protein = None
                max_proteins_per_peptide = None
            elif isinstance(preset, dict):
                min_peptides_per_protein = preset.get("min_peptides_per_protein", min_peptides_per_protein)
                min_unique_peptides_per_protein = preset.get("min_unique_peptides_per_protein", min_unique_peptides_per_protein)
                max_proteins_per_peptide = preset.get("max_proteins_per_peptide", max_proteins_per_peptide)
            else:
                raise ValueError(f"Unknown RS filtering preset: {preset}")

        pdata = self.copy() if return_copy else self # type: ignore[attr-defined], EditingMixin

        rs = pdata.rs # type: ignore[attr-defined]

        # --- Step 1: Peptide filter (max proteins per peptide) ---
        if max_proteins_per_peptide is not None:
            peptide_links = rs.getnnz(axis=0)
            keep_peptides = peptide_links <= max_proteins_per_peptide
            rs = rs[:, keep_peptides]
        else:
            keep_peptides = np.ones(rs.shape[1], dtype=bool)

        # --- Step 2: Protein filters ---
        is_unique = rs.getnnz(axis=0) == 1
        unique_counts = rs[:, is_unique].getnnz(axis=1)
        peptide_counts = rs.getnnz(axis=1)

        keep_proteins = np.ones(rs.shape[0], dtype=bool)
        if min_peptides_per_protein is not None:
            keep_proteins &= (peptide_counts >= min_peptides_per_protein)
        if min_unique_peptides_per_protein is not None:
            keep_proteins &= (unique_counts >= min_unique_peptides_per_protein)

        rs_filtered = rs[keep_proteins, :]

        # --- Step 3: Re-filter peptides now unmapped ---
        keep_peptides_final = rs_filtered.getnnz(axis=0) > 0
        rs_filtered = rs_filtered[:, keep_peptides_final]

        # --- Apply filtered RS ---
        pdata._set_RS(rs_filtered, validate=False) # type: ignore[attr-defined], EditingMixin

        # --- Filter .prot and .pep ---
        if pdata.prot is not None:
            pdata.prot = pdata.prot[:, keep_proteins]
        if pdata.pep is not None:
            original_peptides = keep_peptides.nonzero()[0]
            final_peptides = original_peptides[keep_peptides_final]
            pdata.pep = pdata.pep[:, final_peptides]

        # --- History and summary ---
        n_prot_before = self.prot.shape[1] if self.prot is not None else rs.shape[0]
        n_pep_before = self.pep.shape[1] if self.pep is not None else rs.shape[1]
        n_prot_after = rs_filtered.shape[0]
        n_pep_after = rs_filtered.shape[1]

        n_prot_dropped = n_prot_before - n_prot_after
        n_pep_dropped = n_pep_before - n_pep_after
        
        msg = "🧪 Filtered RS"
        if preset:
            msg += f" using preset '{preset}'"
        if min_peptides_per_protein is not None:
            msg += f", min peptides per protein: {min_peptides_per_protein}"
        if min_unique_peptides_per_protein is not None:
            msg += f", min unique peptides: {min_unique_peptides_per_protein}"
        if max_proteins_per_peptide is not None:
            msg += f", max proteins per peptide: {max_proteins_per_peptide}"
        msg += (
            f". Proteins: {n_prot_before} → {n_prot_after} (dropped {n_prot_dropped}), "
            f"Peptides: {n_pep_before} → {n_pep_after} (dropped {n_pep_dropped})."
        )

        pdata._append_history(msg) # type: ignore[attr-defined], HistoryMixin
        print(msg)
        pdata.update_summary() # type: ignore[attr-defined], SummaryMixin

        # --- Save filter indices to .uns ---
        protein_indices = list(pdata.prot.var_names) if pdata.prot is not None else []
        peptide_indices = list(pdata.pep.var_names) if pdata.pep is not None else []
        pdata.prot.uns['filter_rs'] = {
            "kept_proteins": protein_indices,
            "kept_peptides": peptide_indices,
            "n_proteins": len(protein_indices),
            "n_peptides": len(peptide_indices),
            "description": msg
        }

        if validate_after:
            pdata.validate(verbose=True) # type: ignore[attr-defined], ValidationMixin

        return pdata if return_copy else None

    def _apply_rs_filter(
        self,
        keep_proteins=None,
        keep_peptides=None,
        orig_prot_names=None,
        orig_pep_names=None,
        debug=True
    ):
        """
        Apply filtering to `.prot`, `.pep`, and `.rs` based on protein/peptide masks or name lists.

        This method filters the relational structure (RS matrix) and associated data objects by retaining
        only the specified proteins and/or peptides. Original axis names can be provided to ensure correct
        alignment after prior filtering steps.

        Args:
            keep_proteins (list or np.ndarray or bool array, optional): List of protein names or boolean mask 
                indicating which proteins (RS matrix rows) to keep.
            keep_peptides (list or np.ndarray or bool array, optional): List of peptide names or boolean mask 
                indicating which peptides (RS matrix columns) to keep.
            orig_prot_names (list or np.ndarray, optional): Original protein names corresponding to RS matrix rows.
            orig_pep_names (list or np.ndarray, optional): Original peptide names corresponding to RS matrix columns.
            debug (bool): If True, prints filtering details and index alignment diagnostics.

        Returns:
            None
        """

        if self.rs is None: # type: ignore[attr-defined]
            raise ValueError("No RS matrix to filter.")

        from scipy.sparse import issparse

        rs = self.rs # type: ignore[attr-defined]

        # Use provided names or fallback to current .prot/.pep
        prot_names = np.array(orig_prot_names) if orig_prot_names is not None else np.array(self.prot.var_names)
        pep_names = np.array(orig_pep_names) if orig_pep_names is not None else np.array(self.pep.var_names)

        if rs.shape[0] != len(prot_names) or rs.shape[1] != len(pep_names):
            raise ValueError(
                f"RS shape {rs.shape} does not match provided protein/peptide names "
                f"({len(prot_names)} proteins, {len(pep_names)} peptides). "
                "Did you forget to pass the original names?"
            )

        # --- Normalize protein mask ---
        if keep_proteins is None:
            prot_mask = np.ones(rs.shape[0], dtype=bool)
        elif isinstance(keep_proteins, (list, np.ndarray, pd.Index)) and isinstance(keep_proteins[0], str):
            keep_set = set(keep_proteins)
            prot_mask = np.fromiter((p in keep_set for p in prot_names), dtype=bool)
        elif isinstance(keep_proteins, (list, np.ndarray)) and isinstance(keep_proteins[0], (bool, np.bool_)):
            prot_mask = np.asarray(keep_proteins)
        else:
            raise TypeError("keep_proteins must be a list of str or a boolean mask.")

        # --- Normalize peptide mask ---
        if keep_peptides is None:
            pep_mask = np.ones(rs.shape[1], dtype=bool)
        elif isinstance(keep_peptides, (list, np.ndarray, pd.Index)) and isinstance(keep_peptides[0], str):
            keep_set = set(keep_peptides)
            pep_mask = np.fromiter((p in keep_set for p in pep_names), dtype=bool)
        elif isinstance(keep_peptides, (list, np.ndarray)) and isinstance(keep_peptides[0], (bool, np.bool_)):
            pep_mask = np.asarray(keep_peptides)
        else:
            raise TypeError("keep_peptides must be a list of str or a boolean mask.")

        # --- Final safety check ---
        if len(prot_mask) != rs.shape[0] or len(pep_mask) != rs.shape[1]:
            raise ValueError("Mismatch between mask lengths and RS matrix dimensions.")

        # --- Apply to RS ---
        self._set_RS(rs[prot_mask, :][:, pep_mask], validate=False) # type: ignore[attr-defined], EditingMixin

        # --- Apply to .prot and .pep ---
        kept_prot_names = np.array(orig_prot_names)[prot_mask]
        kept_pep_names = np.array(orig_pep_names)[pep_mask]

        if self.prot is not None:
            self.prot = self.prot[:, self.prot.var_names.isin(kept_prot_names)]

        if self.pep is not None:
            self.pep = self.pep[:, self.pep.var_names.isin(kept_pep_names)]

        if debug:
            print(f"{format_log_prefix('result')} RS matrix filtered: {prot_mask.sum()} proteins, {pep_mask.sum()} peptides retained.")

    def _format_filter_query(self, condition, dataframe):
        """
        Format a condition string for safe evaluation on a DataFrame with complex column names.

        This method prepares a query string for use with `pandas.eval()` or `.query()` by:
        - Wrapping column names containing spaces or special characters in backticks.
        - Converting custom `includes` syntax (for substring matching) into `.str.contains(...)` expressions.
        - Quoting unquoted string values automatically for object/category columns.

        Args:
            condition (str): The condition string to parse and format.
            dataframe (pd.DataFrame): The DataFrame whose column names and dtypes are used for formatting.

        Returns:
            str: A cleaned and properly formatted condition string suitable for `.eval()` or `.query()`.

        Note:
            This is an internal helper used by methods such as `filter_sample_condition()` and `filter_prot()` 
            to support user-friendly string-based queries.
        """

        # Wrap column names with backticks if needed
        column_names = dataframe.columns.tolist()
        column_names.sort(key=len, reverse=True) # Avoid partial matches
        
        for col in column_names:
            if re.search(r'[^\\w]', col):  # Non-alphanumeric characters
                condition = re.sub(fr'(?<!`)({re.escape(col)})(?!`)', f'`{col}`', condition)

        # Handle 'includes' syntax for substring matching
        match = re.search(r'`?(\w[\w\s:.-]*)`?\s+includes\s+[\'"]([^\'"]+)[\'"]', condition)
        if match:
            col_name = match.group(1)
            substring = match.group(2)
            condition = f"{col_name}.str.contains('{substring}', case=False, na=False)"

        # Auto-quote string values for categorical/text columns
        for col in dataframe.columns:
            if dataframe[col].dtype.name in ["object", "category"]:
                for op in ["==", "!="]:
                    pattern = fr"(?<![><=!])\b{re.escape(col)}\s*{op}\s*([^\s'\"()]+)"
                    matches = re.findall(pattern, condition)
                    for match_val in matches:
                        quoted_val = f'"{match_val}"'
                        condition = re.sub(fr"({re.escape(col)}\s*{op}\s*){match_val}\b", r"\1" + quoted_val, condition)

        return condition

    def _annotate_found_samples(self, threshold=0.0, layer='X'):
        """
        Annotate proteins and peptides with per-sample 'Found In' flags.

        For each sample, this method adds boolean indicators to `.prot.var` and `.pep.var` 
        indicating whether the feature was detected (i.e. exceeds the given threshold).

        Args:
            threshold (float): Minimum value to consider a feature as "found" (default is 0.0).
            layer (str): Name of the data layer to evaluate (e.g., "X", "imputed", etc.).

        Returns:
            None

        Note:
            This is an internal helper used to support downstream grouping-based detection 
            (e.g., in `annotate_found_in()`).
            Adds new columns to `.var` of the form: `'Found In: <sample>' → bool`.
        """
        for level in ['prot', 'pep']:
            adata = getattr(self, level)
            # Skip if the level doesn't exist
            if adata is None:
                continue

            # Handle layer selection
            if layer == 'X':
                data = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                                    index=adata.obs_names,
                                    columns=adata.var_names).T
            elif layer in adata.layers:
                data = pd.DataFrame(adata.layers[layer].toarray() if hasattr(adata.layers[layer], 'toarray') else adata.layers[layer],
                                    index=adata.obs_names,
                                    columns=adata.var_names).T
            else:
                raise KeyError(f"Layer '{layer}' not found in {level}.layers and is not 'X'.")

            found = data > threshold
            for sample in found.columns:
                adata.var[f"Found In: {sample}"] = found[sample]

    def annotate_found(self, classes=None, on='protein', layer='X', threshold=0.0, indent=1, verbose=True):
        """
        Add group-level 'Found In' annotations for proteins or peptides.

        This method computes detection flags across groups of samples, based on a minimum intensity 
        threshold, and stores them in `.prot.var` or `.pep.var` as new boolean columns.

        Args:
            classes (str or list of str): Sample-level class/grouping column(s) in `.sample.obs`.
            on (str): Whether to annotate 'protein' or 'peptide' level features.
            layer (str): Data layer to use for evaluation (default is "X").
            threshold (float): Minimum intensity value to be considered "found".
            indent (int): Indentation level for printed messages.
            verbose (bool): If True, prints a summary message after annotation.

        Returns:
            None

        Examples:
            Annotate proteins by group using sample-level metadata:
                ```python
                pdata.annotate_found(classes=["group", "condition"], on="protein")
                ```

            Filter for proteins found in at least 20% of samples from a given group:
                ```python
                pdata_filtered = pdata.filter_prot_found(group="Group_A", min_ratio=0.2)
                pdata_filtered.prot.var
                ```
        """
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            return

        adata = self.prot if on == 'protein' else self.pep
        var = adata.var

        # Handle layer correctly (supports 'X' or adata.layers keys)
        if layer == 'X':
            data = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                                index=adata.obs_names,
                                columns=adata.var_names).T
        elif layer in adata.layers:
            raw = adata.layers[layer]
            data = pd.DataFrame(raw.toarray() if hasattr(raw, 'toarray') else raw,
                                index=adata.obs_names,
                                columns=adata.var_names).T
        else:
            raise KeyError(f"Layer '{layer}' not found in {on}.layers and is not 'X'.")

        found_df = data > threshold

        # Prepare or retrieve existing numeric storage in .uns
        metrics_key = f"found_metrics_{on}"
        metrics_df = adata.uns.get(metrics_key, pd.DataFrame(index=adata.var_names))

        if classes is not None:
            classes_list = utils.get_classlist(adata, classes=classes)

            for class_value in classes_list:
                class_data = utils.resolve_class_filter(adata, classes, class_value)
                class_samples = class_data.obs_names

                if len(class_samples) == 0:
                    continue

                sub_found = found_df[class_samples]
                count = sub_found.sum(axis=1)
                ratio = count / len(class_samples)

                # Store display-friendly annotations in .var
                var[f"Found In: {class_value}"] = sub_found.any(axis=1)
                var[f"Found In: {class_value} ratio"] = sub_found.sum(axis=1).astype(str) + "/" + str(len(class_samples))
                
                # Store numeric data in .uns
                metrics_df[(class_value, "count")] = count
                metrics_df[(class_value, "ratio")] = ratio

            # Store updated versions back into .uns
            metrics_df.columns = pd.MultiIndex.from_tuples(metrics_df.columns)
            metrics_df = metrics_df.sort_index(axis=1)
            adata.uns[metrics_key] = metrics_df

        self._history.append(  # type: ignore[attr-defined], HistoryMixin
            f"{on}: Annotated features 'found in' class combinations {classes} using threshold {threshold}."
        )
        if verbose:
            print(
                f"{format_log_prefix('user', indent=indent)} Annotated 'found in' features by group: "
                f"{classes} using threshold {threshold}."
            )

    def annotate_significant(self, classes=None, on='protein', fdr_threshold=0.01, indent=1, verbose=True):
        """
        Add group-level 'Significant In' annotations for proteins or peptides.

        This method computes significance flags (e.g., X_qval < threshold) across groups 
        of samples and stores them in `.prot.var` or `.pep.var` as new boolean columns.

        DIA-NN: protein X_qval originates from PG.Q.Value, peptide X_qval originates from Q.Value.

        Args:
            classes (str or list of str, optional): Sample-level grouping column(s) for group-based annotations.
            fdr_threshold (float): Significance threshold (default 0.01).
            on (str): Level to annotate ('protein' or 'peptide').
            indent (int): Indentation level for printed messages.
            verbose (bool): If True, prints a summary message after annotation.

        Returns:
            None

        Examples:
            Annotate proteins by group using sample-level metadata:
                ```python
                pdata.annotate_significant(classes="celltype", on="protein", fdr_threshold=0.01)
                ```
        """
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            return

        adata = self.prot if on == 'protein' else self.pep
        var = adata.var

        # Check if significance layer exists
        sig_layer = 'X_qval'
        if sig_layer not in adata.layers:
            # Try global q-values as fallback
            if "Global_Q_value" in var.columns:
                global_mask = var["Global_Q_value"] < fdr_threshold
                var["Significant In: Global"] = global_mask
                adata.uns[f"significance_metrics_{on}"] = pd.DataFrame(
                    {"Global_count": global_mask.sum(), "Global_ratio": global_mask.mean()},
                    index=["Global"]
                )
                if verbose:
                    print(f"{format_log_prefix('user', indent=indent)} Annotated global significant features using FDR threshold {fdr_threshold}.")
                return
            else:
                raise ValueError(
                    f"No per-sample layer ('{sig_layer}') or 'Global_Q_value' column found for {on}-level significance."
                )

        data = pd.DataFrame(
            adata.layers[sig_layer].toarray() if hasattr(adata.layers[sig_layer], 'toarray') else adata.layers[sig_layer],
            index=adata.obs_names,
            columns=adata.var_names).T  # shape: features × samples
        
        # Group-level summary
        metrics_key = f"significance_metrics_{on}"
        metrics_df = adata.uns.get(metrics_key, pd.DataFrame(index=adata.var_names))

        if classes is not None:
            classes_list = utils.get_classlist(adata, classes=classes)

            for class_value in classes_list:
                class_data = utils.resolve_class_filter(adata, classes, class_value)
                class_samples = class_data.obs_names

                if len(class_samples) == 0:
                    continue

                sub_df = data[class_samples]
                count = (sub_df < fdr_threshold).sum(axis=1)
                ratio = count / len(class_samples)

                var[f"Significant In: {class_value}"] = (sub_df < fdr_threshold).any(axis=1)
                var[f"Significant In: {class_value} ratio"] = count.astype(str) + "/" + str(len(class_samples))

                metrics_df[(class_value, "count")] = count
                metrics_df[(class_value, "ratio")] = ratio

            metrics_df.columns = pd.MultiIndex.from_tuples(metrics_df.columns)
            metrics_df = metrics_df.sort_index(axis=1)
            adata.uns[metrics_key] = metrics_df

        self._history.append(  # type: ignore[attr-defined], HistoryMixin
            f"{on}: Annotated significance across classes {classes} using FDR threshold {fdr_threshold}."
        )
        if verbose:
            print(
                f"{format_log_prefix('user', indent=indent)} Annotated significant features by group: {classes} using FDR threshold {fdr_threshold}."
            )

    def _annotate_significant_samples(self, fdr_threshold=0.01):
        """
        Annotate proteins and peptides with per-sample 'Significant In' flags.

        For each sample, this method adds boolean indicators to `.prot.var` and `.pep.var`
        indicating whether the feature is significantly detected (i.e. FDR < threshold).

        Args:
            fdr_threshold (float): FDR threshold for significance (default: 0.01).

        Returns:
            None

        Note:
            This is an internal helper used to support downstream grouping-based annotation.
            Adds new columns to `.var` of the form: `'Significant In: <sample>' → bool`.
        """
        for level in ['prot', 'pep']:
            adata = getattr(self, level)
            if adata is None:
                continue

            if "Global_Q_value" in adata.var.columns:
                adata.var["Significant In: Global"] = adata.var["Global_Q_value"] < fdr_threshold

            sig_layer = 'X_qval'
            if sig_layer not in adata.layers:
                # Fallback to global q-values if available
                print(f"{format_log_prefix('info', 2)} Using global q-values for '{level}' significance annotation.")
                continue  # Skip if significance data not available

            print(f"{format_log_prefix('info', 2)} Using sampl-specific q-values for '{level}' significance annotation.")

            data = pd.DataFrame(
                adata.layers[sig_layer].toarray() if hasattr(adata.layers[sig_layer], 'toarray') else adata.layers[sig_layer],
                index=adata.obs_names,
                columns=adata.var_names).T  # features × samples

            significant = data < fdr_threshold
            sig_df = significant.add_prefix("Significant In: ")
            adata.var = pd.concat([adata.var, sig_df], axis=1)
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import variation, ttest_ind, mannwhitneyu, wilcoxon
from scviz import utils
from scviz.utils import format_log_prefix
import warnings
from scipy import sparse


class AnalysisMixin:
    """
    Statistical and dimensionality reduction methods for bulk and single-cell proteomics.

    Functions:
        cv: Computes coefficient of variation.
        de: Runs differential expression analysis.
        rank: Ranks features using group separation metrics.
        impute: Fills in missing values using KNN or class-based strategies.
        normalize: Row-wise normalization across proteins or peptides.
        _normalize_helper: Internal logic for normalization routines.
        clean_X: NaN or outlier clean-up of `.X`.
        namissingvalues: Counts missing values across samples or features.
        neighbor: Builds neighbor graph using Scanpy.
        leiden: Runs community detection on the neighbor graph.
        umap: Performs UMAP dimensionality reduction.
        pca: Principal component analysis.
    """
    def cv(self, classes = None, on = 'protein', layer = "X", debug = False):
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass

        adata = self.prot if on == 'protein' else self.pep
        classes_list = utils.get_classlist(adata, classes)

        for j, class_value in enumerate(classes_list):
            data_filtered = utils.resolve_class_filter(adata, classes, class_value, debug=True)

            cv_data = data_filtered.X.toarray() if layer == "X" else data_filtered.layers[layer].toarray() if layer in data_filtered.layers else None
            if cv_data is None:
                raise ValueError(f"Layer '{layer}' not found in adata.layers.")

            adata.var['CV: '+ class_value] = variation(cv_data, axis=0)

        self._history.append(f"{on}: Coefficient of Variation (CV) calculated for {layer} data by {classes}. E.g. CV stored in var['CV: {class_value}'].") # type: ignore[attr-defined]

    # TODO: implement methods for calculdating fold change, 1. mean, 2. prot pairwise median, or 3. pep pairwise median (will need to refer to RS)
    def de(self, values=None, class_type=None, method='ttest', layer='X', pval=0.05, log2fc=1.0, fold_change_mode='mean'):
        """
        Calculate differential expression (DE) of proteins across different groups.

        This function calculates the DE of proteins across different groups. The cases to compare can be specified, and the method to use for DE can be specified as well.
        Supports legacy (class_type + values) or new dictionary-style filtering.

        Parameters:
        values : list of dict or list of list
            Two sample group filters to compare. Each group should be either:
            - A dictionary of {class: value} (e.g., {'cellline': 'HCT116', 'treatment': 'DMSO'})
            - A list of values (legacy-style) if `class_type` is provided
        class_type : str or list of str, optional
            Legacy-style class label(s). Used only if values is a list of value combinations.
        method : str
            Statistical test to use. Options: 'ttest', 'mannwhitneyu', 'wilcoxon'.
        layer : str
            Data layer to use for DE (e.g., "X", "X_raw", "X_mbr").
        pval : float
            P-value cutoff for significance labeling.
        log2fc : float
            Log2 fold change threshold for labeling.
        fold_change_mode : str
            Method for computing fold change. Options:
            - 'mean' : log2(mean(group1) / mean(group2))
            - 'pairwise_median' : median of all pairwise log2 ratios
            - 'pep_pairwise_median' : median of all pairwise log2 ratios for peptides per protein

        Returns:
        df_stats : pandas.DataFrame
            A DataFrame containing log2 fold change, p-values, and significance labels for each protein.

        Examples:
        # Legacy-style usage
        >>> pdata.de(class_type=['cellline', 'treatment'],
        ...          values=[['HCT116', 'DMSO'], ['HCT116', 'DrugX']])

        # Dictionary-style usage (recommended)
        >>> pdata.de(values=[
        ...     {'cellline': 'HCT116', 'treatment': 'DMSO'},
        ...     {'cellline': 'HCT116', 'treatment': 'DrugX'}
        ... ])
        """

        # --- Handle legacy input ---
        if values is None:
            raise ValueError("Please provide `values` (new format) or both `class_type` and `values` (legacy format).")

        if isinstance(values, list) and all(isinstance(v, list) for v in values) and class_type is not None:
            values = utils.format_class_filter(class_type, values, exact_cases=True)

        if not isinstance(values, list) or len(values) != 2:
            raise ValueError("`values` must be a list of two group dictionaries (or legacy value pairs).")

        group1_dict, group2_dict = (
            [values[0]] if not isinstance(values[0], list) else values[0],
            [values[1]] if not isinstance(values[1], list) else values[1]
        )


        # --- Sample filtering ---
        pdata_case1 = self._filter_sample_values(values=group1_dict, exact_cases=True, return_copy=True, verbose=False) # type: ignore[attr-defined], FilteringMixin
        pdata_case2 = self._filter_sample_values(values=group2_dict, exact_cases=True, return_copy=True, verbose=False) # type: ignore[attr-defined], FilteringMixin

        def _label(d):
            if isinstance(d, dict):
                return '_'.join(str(v) for v in d.values())
            return str(d)

        group1_string = _label(group1_dict)
        group2_string = _label(group2_dict)
        comparison_string = f'{group1_string} vs {group2_string}'

        log_prefix = format_log_prefix("user")
        n1, n2 = len(pdata_case1.prot), len(pdata_case2.prot)
        print(f"{log_prefix} Running differential expression [protein]")
        print(f"   🔸 Comparing groups: {comparison_string}")
        print(f"   🔸 Group sizes: {n1} vs {n2} samples")
        print(f"   🔸 Method: {method} | Fold Change: {fold_change_mode} | Layer: {layer}")
        print(f"   🔸 P-value threshold: {pval} | Log2FC threshold: {log2fc}")

        # --- Get layer data ---
        data1 = utils.get_adata_layer(pdata_case1.prot, layer)
        data2 = utils.get_adata_layer(pdata_case2.prot, layer)

        # Shape: (samples, features)
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)

        # --- Compute fold change ---
        if fold_change_mode == 'mean':
            group1_mean = np.nanmean(data1, axis=0)
            group2_mean = np.nanmean(data2, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                log2fc_vals = np.log2(group1_mean / group2_mean)

        elif fold_change_mode == 'pairwise_median':
            log2fc_vals = utils.pairwise_log2fc(data1, data2)
        
        elif fold_change_mode == 'pep_pairwise_median':
            # --- Validate .pep presence ---
            if self.pep is None:
                raise ValueError("Peptide-level data (.pep) is required for fold_change_mode='pep_pairwise_median', but self.pep is None.")

            # --- Handle peptide layer fallback ---
            actual_layer = layer
            if layer != 'X' and not (hasattr(self.pep, "layers") and layer in self.pep.layers):
                warnings.warn(
                    f"Layer '{layer}' not found in .pep.layers. Falling back to 'X'.",
                    UserWarning
                )
                actual_layer = 'X'

            # Get peptide data
            pep_data1 = utils.get_adata_layer(pdata_case1.pep, actual_layer)
            pep_data2 = utils.get_adata_layer(pdata_case2.pep, actual_layer)
            pep_data1 = np.asarray(pep_data1)
            pep_data2 = np.asarray(pep_data2)

            # Compute per-peptide pairwise log2FCs
            pep_log2fc = utils.pairwise_log2fc(pep_data1, pep_data2)

            # Map peptides to proteins
            pep_to_prot = utils.get_pep_prot_mapping(self, return_series=True)

            # Aggregate peptide log2FCs into protein-level log2FCs
            prot_log2fc = pd.Series(index=self.prot.var_names, dtype=float)
            for prot in self.prot.var_names:
                matching_peptides = pep_to_prot[pep_to_prot == prot].index
                if len(matching_peptides) == 0:
                    continue
                idxs = self.pep.var_names.get_indexer(matching_peptides)
                valid_idxs = idxs[idxs >= 0]
                if len(valid_idxs) > 0:
                    prot_log2fc[prot] = np.nanmedian(pep_log2fc[valid_idxs])
            log2fc_vals = prot_log2fc.values
        else:
            raise ValueError(f"Unsupported fold_change_mode: {fold_change_mode}")

        # --- Statistical test ---
        pvals = []
        stats = []
        for i in range(data1.shape[1]):
            x1, x2 = data1[:, i], data2[:, i]
            try:
                if method == 'ttest':
                    res = ttest_ind(x1, x2, nan_policy='omit')
                elif method == 'mannwhitneyu':
                    res = mannwhitneyu(x1, x2, alternative='two-sided')
                elif method == 'wilcoxon':
                    res = wilcoxon(x1, x2)
                else:
                    raise ValueError(f"Unsupported test method: {method}")
                pvals.append(res.pvalue)
                stats.append(res.statistic)
            except Exception as e:
                pvals.append(np.nan)
                stats.append(np.nan)

        # --- Compile results ---
        var = self.prot.var.copy()
        df_stats = pd.DataFrame(index=self.prot.var_names)
        df_stats['Genes'] = var['Genes'] if 'Genes' in var.columns else var.index
        df_stats[group1_string] = np.nanmean(data1, axis=0)
        df_stats[group2_string] = np.nanmean(data2, axis=0)
        df_stats['log2fc'] = log2fc_vals
        df_stats['p_value'] = pvals
        df_stats['test_statistic'] = stats

        df_stats['-log10(p_value)'] = -np.log10(df_stats['p_value'].replace(0, np.nan).astype(float))
        df_stats['significance_score'] = df_stats['-log10(p_value)'] * df_stats['log2fc']
        df_stats['significance'] = 'not significant'
        df_stats.loc[(df_stats['p_value'] < pval) & (df_stats['log2fc'] > log2fc), 'significance'] = 'upregulated'
        df_stats.loc[(df_stats['p_value'] < pval) & (df_stats['log2fc'] < -log2fc), 'significance'] = 'downregulated'
        df_stats['significance'] = pd.Categorical(df_stats['significance'], categories=['upregulated', 'downregulated', 'not significant'], ordered=True)

        df_stats = df_stats.dropna(subset=['p_value', 'log2fc', 'significance'])
        df_stats = df_stats.sort_values(by='significance')

        # --- Store and return ---
        self._stats[comparison_string] = df_stats # type: ignore[attr-defined]
        self._append_history(f"prot: DE for {class_type} {values} using {method} and fold_change_mode='{fold_change_mode}'. Stored in .stats['{comparison_string}'].") # type: ignore[attr-defined], HistoryMixin

        sig_counts = df_stats['significance'].value_counts().to_dict()
        n_up = sig_counts.get('upregulated', 0)
        n_down = sig_counts.get('downregulated', 0)
        n_ns = sig_counts.get('not significant', 0)

        print(f"{format_log_prefix('result_only', indent=2)} DE complete. Results stored in:")
        print(f"       • .stats['{comparison_string}']")
        print(f"       • Columns: log2fc, p_value, significance, etc.")
        print(f"       • Upregulated: {n_up} | Downregulated: {n_down} | Not significant: {n_ns}")

        return df_stats

    # TODO: Need to figure out how to make this interface with plot functions, probably do reordering by each class_value within the loop?
    def rank(self, classes = None, on = 'protein', layer = "X"):
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass

        adata = self.prot if on == 'protein' else self.pep
        classes_list = utils.get_classlist(adata, classes)
        
        for j, class_value in enumerate(classes_list):
            rank_data = utils.resolve_class_filter(adata, classes, class_value, debug=True)

            rank_df = rank_data.to_df().transpose()
            rank_df['Average: '+class_value] = np.nanmean(rank_data.X.toarray(), axis=0)
            rank_df['Stdev: '+class_value] = np.nanstd(rank_data.X.toarray(), axis=0)
            rank_df.sort_values(by=['Average: '+class_value], ascending=False, inplace=True)
            rank_df['Rank: '+class_value] = np.where(rank_df['Average: '+class_value].isna(), np.nan, np.arange(1, len(rank_df) + 1))

            # revert back to original data order (need to do so, because we sorted rank_df)
            sorted_indices = rank_df.index
            rank_df = rank_df.loc[adata.var.index]
            adata.var['Average: ' + class_value] = rank_df['Average: ' + class_value]
            adata.var['Stdev: ' + class_value] = rank_df['Stdev: ' + class_value]
            adata.var['Rank: ' + class_value] = rank_df['Rank: ' + class_value]
            rank_df = rank_df.reindex(sorted_indices)

        self._history.append(f"{on}: Ranked {layer} data. Ranking, average and stdev stored in var.") # type: ignore[attr-defined], HistoryMixin

    def impute(self, classes=None, layer="X", method='mean', on='protein', set_X=True, **kwargs):
        """
        Impute missing values across samples (globally or within classes) using SimpleImputer.

        Parameters:
            classes (str or list): Class columns in .obs to group by.
            layer (str): Data layer to impute from.
            method (str): 'mean', 'median', or 'min'.
            on (str): 'protein' or 'peptide'.
            set_X (bool): Whether to set .X to the imputed result.
        """
        from sklearn.impute import SimpleImputer, KNNImputer
        from scipy import sparse
        from scviz import utils


        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            return

        adata = self.prot if on == 'protein' else self.pep
        if layer != "X" and layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in .{on}.")

        impute_data = adata.layers[layer] if layer != "X" else adata.X
        was_sparse = sparse.issparse(impute_data)
        impute_data = impute_data.toarray() if was_sparse else impute_data.copy()
        original_data = impute_data.copy()

        layer_name = f"X_impute_{method}"

        if method not in {"mean", "median", "min","knn"}:
            raise ValueError(f"Unsupported method: {method}")

        if classes is None:
            # Global imputation
            if method == 'min':
                min_vals = np.nanmin(impute_data, axis=0)
                min_vals = np.where(np.isnan(min_vals), 0, min_vals)
                mask = np.isnan(impute_data)
                impute_data[mask] = np.take(min_vals, np.where(mask)[1])
            elif method == 'knn':
                n_neighbors = kwargs.get('n_neighbors', 3)
                imputer = KNNImputer(n_neighbors=n_neighbors)
                impute_data = imputer.fit_transform(impute_data)
            else:
                imputer = SimpleImputer(strategy=method, keep_empty_features=True)
                impute_data = imputer.fit_transform(impute_data)

            print(f"{format_log_prefix('user')} Global imputation using '{method}'. Layer saved as '{layer_name}'.")

        else:
            # Group-wise imputation
            if method == 'knn':
                raise ValueError("KNN imputation is not supported for group-wise imputation.")

            sample_names = utils.get_samplenames(adata, classes)
            sample_names = np.array(sample_names)
            unique_groups = np.unique(sample_names)

            for group in unique_groups:
                idx = np.where(sample_names == group)[0]
                group_data = impute_data[idx, :]

                if method == 'min':
                    min_vals = np.nanmin(group_data, axis=0)
                    min_vals = np.where(np.isnan(min_vals), 0, min_vals)
                    mask = np.isnan(group_data)
                    group_data[mask] = np.take(min_vals, np.where(mask)[1])
                    imputed_group = group_data
                else:
                    imputer = SimpleImputer(strategy=method, keep_empty_features=True)
                    imputed_group = imputer.fit_transform(group_data)

                impute_data[idx, :] = imputed_group

            print(f"{format_log_prefix('user')} Group-wise imputation using '{method}' on class(es): {classes}. Layer saved as '{layer_name}'.")

        summary_lines = []
        if classes is None:
            num_imputed = np.sum(np.isnan(original_data) & ~np.isnan(impute_data))
            # Row-wise missingness
            was_missing = np.isnan(original_data).any(axis=1)
            now_complete = ~np.isnan(impute_data).any(axis=1)
            now_incomplete = np.isnan(impute_data).any(axis=1)

            fully_imputed_samples = np.sum(was_missing & now_complete)
            partially_imputed_samples = np.sum(was_missing & now_incomplete)

            summary_lines.append(
                f"{format_log_prefix('result_only', indent=2)} {num_imputed} values imputed."
            )
            summary_lines.append(
                f"{format_log_prefix('info_only', indent=2)} {fully_imputed_samples} samples fully imputed, {partially_imputed_samples} samples partially imputed."
            )

        else:
            sample_names = utils.get_samplenames(adata, classes)
            sample_names = np.array(sample_names)
            unique_groups = np.unique(sample_names)

            counts_by_group = {}
            fully_by_group = {}
            partial_by_group = {}
            
            for group in unique_groups:



                idx = np.where(sample_names == group)[0]
                before = original_data[idx, :]
                after = impute_data[idx, :]

                # count imputed values
                mask = np.isnan(before) & ~np.isnan(after)
                counts_by_group[group] = np.sum(mask)

                # count fully and partially imputed samples
                was_missing = np.isnan(before).any(axis=1)
                now_complete = ~np.isnan(after).any(axis=1)
                now_incomplete = np.isnan(after).any(axis=1)

                fully_by_group[group] = np.sum(was_missing & now_complete)
                partial_by_group[group] = np.sum(was_missing & now_incomplete)

            total = sum(counts_by_group.values())
            summary_lines.append(f"{format_log_prefix('result_only', indent=2)} {total} values imputed total.")
            for group in unique_groups:
                count = counts_by_group[group]
                fully = fully_by_group[group]
                partial = partial_by_group[group]
                summary_lines.append(
                    f"   - {group}: {count} values, {fully} fully imputed, {partial} partially imputed samples"
                )

        print("\n".join(summary_lines))

        adata.layers[layer_name] = sparse.csr_matrix(impute_data) if was_sparse else impute_data

        if set_X:
            self.set_X(layer=layer_name, on=on) # type: ignore[attr-defined], EditingMixin

        self._history.append( # type: ignore[attr-defined]
            f"{on}: Imputed layer '{layer}' using '{method}' (grouped by {classes if classes else 'ALL'}). Stored in '{layer_name}'."
        )

    def neighbor(self, on = 'protein', layer = "X", **kwargs):
        # uses sc.pp.neighbors
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass
        
        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        if layer == "X":
            # do nothing
            pass
        elif layer in adata.layers.keys():
            self.set_X(layer = layer, on = on) # type: ignore[attr-defined], EditingMixin

        log_prefix = format_log_prefix("user")
        print(f"{log_prefix} Computing neighbors [{on}] using layer: {layer}")

        if 'pca' not in adata.uns:
            print(f"{format_log_prefix('info_only',indent=2)} PCA not found in AnnData object. Running PCA with default settings.")
            self.pca(on = on, layer = layer)

        sc.pp.neighbors(adata, **kwargs)

        self._append_history(f'{on}: Neighbors fitted on {layer}, stored in obs["distances"] and obs["connectivities"]') # type: ignore[attr-defined], HistoryMixin
        print(f"{format_log_prefix('result_only',indent=2)} Neighbors computed on {layer}. Results stored in:")
        print(f"       • obs['distances'] (pairwise distances)")
        print(f"       • obs['connectivities'] (connectivity graph)")
        print(f"       • uns['neighbors'] (neighbor graph metadata)")
 
    def leiden(self, on = 'protein', layer = "X", **kwargs):
        # uses sc.tl.leiden with default resolution of 0.25
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass

        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        log_prefix = format_log_prefix("user")
        print(f"{log_prefix} Performing Leiden clustering [{on}] using layer: {layer}")

        if 'neighbors' not in adata.uns:
            print(f"{format_log_prefix('info_only', indent=2)} Neighbors not found in AnnData object. Running neighbors with default settings.")
            self.neighbor(on = on, layer = layer)

        if layer == "X":
            # do nothing
            pass
        elif layer in adata.layers.keys():
            self.set_X(layer = layer, on = on) # type: ignore[attr-defined], EditingMixin

        sc.tl.leiden(adata, **kwargs)

        self._append_history(f'{on}: Leiden clustering fitted on {layer}, stored in obs["leiden"]') # type: ignore[attr-defined], HistoryMixin
        print(f"{format_log_prefix('result_only', indent=2)} Leiden clustering complete. Results stored in:")
        print(f"       • obs['leiden'] (cluster labels)")

    def umap(self, on = 'protein', layer = "X", **kwargs):
        # uses sc.tl.umap
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass
       
        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        log_prefix = format_log_prefix("user")
        print(f"{log_prefix} Computing UMAP [{on}] using layer: {layer}")

        # check if neighbor has been run before, look for distances and connectivities in obsp
        if 'neighbors' not in adata.uns:
            print(f"{format_log_prefix('info_only', indent=2)} Neighbors not found in AnnData object. Running neighbors with default settings.")
            self.neighbor(on = on, layer = layer)

        if layer == "X":
            # do nothing
            pass
        elif layer in adata.layers.keys():
            self.set_X(layer = layer, on = on) # type: ignore[attr-defined], EditingMixin

        sc.tl.umap(adata, **kwargs)

        self._append_history(f'{on}: UMAP fitted on {layer}, stored in obsm["X_umap"] and uns["umap"]') # type: ignore[attr-defined], HistoryMixin
        print(f"{format_log_prefix('result_only', indent=2)} UMAP complete. Results stored in:")
        print(f"       • obsm['X_umap'] (UMAP coordinates)")
        print(f"       • uns['umap'] (UMAP settings)")

    def pca(self, on = 'protein', layer = "X", **kwargs):
        # uses sc.tl.pca
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass
        
        if on == 'protein':
            adata = self.prot
        elif on == 'peptide':
            adata = self.pep

        # make sample array
        if layer == "X":
            X = adata.X.toarray()
        elif layer in adata.layers.keys():
            X = adata.layers[layer].toarray()

        log_prefix = format_log_prefix("user")
        print(f"{log_prefix} Performing PCA [{on}] using layer: {layer}, removing NaN features.")
        print(f"   🔸 BEFORE (samples × proteins): {X.shape}")
        Xnorm = (X - X.mean(axis=0)) / X.std(axis=0)
        nan_cols = np.isnan(Xnorm).any(axis=0)
        Xnorm = Xnorm[:, ~nan_cols]
        print(f"   🔸 AFTER  (samples × proteins): {Xnorm.shape}")

        pca_data = sc.tl.pca(Xnorm, return_info=True, **kwargs)
        adata.obsm['X_pca'] = pca_data[0]
        PCs = np.zeros((pca_data[1].shape[0], nan_cols.shape[0]))
        
        # fill back the 0s where column was NaN in the original data, and thus not used in PCA
        counter = 0
        for i in range(PCs.shape[1]):
            if not nan_cols[i]:
                PCs[:, i] = pca_data[1][:, counter]
                counter += 1

        adata.uns['pca'] = {'PCs': PCs, 'variance_ratio': pca_data[2], 'variance': pca_data[3]}
        
        subpdata = "prot" if on == 'protein' else "pep"

        self._append_history(f'{on}: PCA fitted on {layer}, stored in obsm["X_pca"] and varm["PCs"]') # type: ignore[attr-defined], HistoryMixin
        print(f"{format_log_prefix('result_only',indent=2)} PCA complete, fitted on {layer}. Results stored in:")
        print(f"       • .{subpdata}.obsm['X_pca']")
        print(f"       • .{subpdata}.uns['pca'] (includes PCs, variance, variance ratio)")
        var_pc1, var_pc2 = pca_data[2][:2]
        print(f"       • Variance explained by PC1/PC2: {var_pc1*100:.2f}% , {var_pc2*100:.2f}%") 

    def nanmissingvalues(self, on = 'protein', limit = 0.5):
        # sets columns (proteins and peptides) with > limit (default 0.5) missing values to NaN across all samples
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            pass

        if on == 'protein':
            adata = self.prot

        elif on == 'peptide':
            adata = self.pep

        missing_proportion = np.isnan(adata.X.toarray()).mean(axis=0)
        columns_to_nan = missing_proportion > limit
        adata.X[:, columns_to_nan] = np.nan

        if on == 'protein':
            self.prot = adata
        elif on == 'peptide':
            self.pep = adata

    def normalize(self, classes = None, layer = "X", method = 'sum', on = 'protein', set_X = True, force = False, use_nonmissing = False, **kwargs):  
        """ 
        Normalize the data across samples (globally or within groups).

        Parameters:
        - classes (str or list): Sample-level class/grouping column(s) in .obs.
        - layer (str): Data layer to normalize from (default='X').
        - method (str): Normalization method. Options: 'sum', 'median', 'mean', 'max', 'reference_feature', 'robust_scale', 'quantile_transform'.
        - on (str): 'protein' or 'peptide'.
        - set_X (bool): Whether to set .X to the normalized result.
        - force (bool): Whether to force normalization even with bad rows.
        - use_nonmissing (bool): If True, only use columns with no missing values across all samples when computing scaling factors.
        - **kwargs: Additional arguments for normalization methods.
            (e.g., reference_columns for 'reference_feature', n_neighbors for 'knn').
            max_missing_fraction: Maximum fraction of missing values allowed in a row. Default is 0.5.

        """
        
        if not self._check_data(on): # type: ignore[attr-defined], ValidationMixin
            return

        adata = self.prot if on == 'protein' else self.pep
        if layer != "X" and layer not in adata.layers:
            raise ValueError(f"Layer {layer} not found in .{on}.")
       
        normalize_data = adata.layers[layer] if layer != "X" else adata.X
        was_sparse = sparse.issparse(normalize_data)
        normalize_data = normalize_data.toarray() if was_sparse else normalize_data.copy()
        original_data = normalize_data.copy()

        # Check for bad rows (too many missing values)
        missing_fraction = np.isnan(normalize_data).sum(axis=1) / normalize_data.shape[1]
        max_missing_fraction = kwargs.pop("max_missing_fraction", 0.5)
        bad_rows_mask = missing_fraction > max_missing_fraction

        if np.any(bad_rows_mask):
            n_bad = np.sum(bad_rows_mask)
            print(f"⚠️ {n_bad} sample(s) have >{int(max_missing_fraction*100)}% missing values.")
            print("   Suggest running `.impute()` before normalization for more stable results.")
            print("   Alternatively, try `use_nonmissing=True` to normalize using only consistently observed proteins.")
            if not force:
                print("   ➡️ Use `force=True` to proceed anyway.")
                return

        layer_name = 'X_norm_' + method
        normalize_funcs = ['sum', 'median', 'mean', 'max', 'reference_feature', 'robust_scale', 'quantile_transform']

        if method not in normalize_funcs:
            raise ValueError(f"Unsupported normalization method: {method}")

        if classes is None:
            normalize_data = self._normalize_helper(normalize_data, method, use_nonmissing=use_nonmissing, **kwargs)
            msg=f"{format_log_prefix('user')} Global normalization using '{method}'"
        else:
            # Group-wise normalization
            sample_names = utils.get_samplenames(adata, classes)
            sample_names = np.array(sample_names)
            unique_groups = np.unique(sample_names)

            for group in unique_groups:
                idx = np.where(sample_names == group)[0]
                group_data = normalize_data[idx, :]

                normalized_group = self._normalize_helper(group_data, method=method, use_nonmissing=use_nonmissing, **kwargs)
                normalize_data[idx, :] = normalized_group

            msg=f"{format_log_prefix('info_only')} Group-wise normalization using '{method}' on class(es): {classes}"

        if use_nonmissing and method in {'sum', 'mean', 'median', 'max'}:
            msg += f" (using only fully observed columns)"

        msg += f". Layer saved as '{layer_name}'."
        print(msg)

        # summary printout
        summary_lines = []
        if classes is None:
            summary_lines.append(f"{format_log_prefix('result_only', indent=2)} Normalized all {normalize_data.shape[0]} samples.")
        else:
            for group in unique_groups:
                count = np.sum(sample_names == group)
                summary_lines.append(f"   - {group}: {count} samples normalized")
            summary_lines.insert(0, f"{format_log_prefix('result_only', indent=2)} Normalized {normalize_data.shape[0]} samples total.")
        print("\n".join(summary_lines))

        adata.layers[layer_name] = sparse.csr_matrix(normalize_data) if was_sparse else normalize_data

        if set_X:
            self.set_X(layer = layer_name, on = on) # type: ignore[attr-defined], EditingMixin

        # Determine if use_nonmissing note should be added
        note = ""
        if use_nonmissing and method in {'sum', 'mean', 'median', 'max'}:
            note = " (using only fully observed columns)"

        self._history.append( # type: ignore[attr-defined], HistoryMixin
            f"{on}: Normalized layer {layer} using {method}{note} (grouped by {classes}). Stored in `{layer_name}`."
            )
    
    def _normalize_helper(self, data, method, use_nonmissing, **kwargs):
        """
        Helper function for row-wise normalization.

        Parameters:
        - data (np.ndarray): Data matrix (samples x features).
        - method (str): Normalization method. One of:
            'sum', 'mean', 'median', 'max', 'reference_feature',
            'robust_scale', 'quantile_transform'.
        - use_nonmissing (bool): If True, only use columns with no missing values
                                across all samples when computing scaling factors.

        Returns:
        - np.ndarray: Normalized data.
        """

        if method in {'sum', 'mean', 'median', 'max'}:
            reducer = {
                    'sum': np.nansum,
                    'mean': np.nanmean,
                    'median': np.nanmedian,
                    'max': np.nanmax
                }[method]

            if use_nonmissing:
                fully_observed_cols = ~np.isnan(data).any(axis=0)
                if not np.any(fully_observed_cols):
                    raise ValueError("No fully observed columns available for normalization with `use_nonmissing=True`.")
                used_cols = np.where(fully_observed_cols)[0]
                print(f"ℹ️ Normalizing using only fully observed columns: {used_cols}")
                row_vals = reducer(data[:, fully_observed_cols], axis=1)
            else:
                row_vals = reducer(data, axis=1)

            scale = np.nanmax(row_vals) / row_vals
            scale = np.where(np.isnan(scale), 1.0, scale)
            data_norm = data * scale[:, None]

        elif method == 'reference_feature':
            # norm by reference feature: scale each row s.t. the reference column is the same across all rows (scale to max value of reference column)
            reference_columns = kwargs.get('reference_columns', [2])
            reference_method = kwargs.get('reference_method', 'median')  # default to median

            reducer_map = {
                'mean': np.nanmean,
                'median': np.nanmedian,
                'sum': np.nansum
            }

            if reference_method not in reducer_map:
                raise ValueError(f"Unsupported reference method: {reference_method}. Supported methods are: {list(reducer_map.keys())}")
            reducer = reducer_map[reference_method]

            # resolve reference column names if needed
            if isinstance(reference_columns[0], str):
                gene_to_acc, _ = self.get_gene_maps(on='protein') # type: ignore[attr-defined], IdentifierMixin
                resolved = utils.resolve_accessions(self.prot, reference_columns, gene_map=gene_to_acc)
                reference_acc = [ref for ref in resolved if ref in self.prot.var.index]
                reference_columns = [self.prot.var.index.get_loc(ref) for ref in reference_acc]
                print(f"{format_log_prefix('info')} Normalizing using found reference columns: {reference_acc}")
                self._history.append(f"Used reference_feature normalization with resolved accessions: {resolved}") # type: ignore[attr-defined]
            else:
                reference_columns = [int(ref) for ref in reference_columns]
                reference_acc = [self.prot.var.index[ref] for ref in reference_columns if ref < self.prot.shape[1]]
                print(f"{format_log_prefix('info')} Normalizing using reference columns: {reference_acc}")
                self._history.append(f"Used reference_feature normalization with resolved accessions: {reference_acc}") # type: ignore[attr-defined]

            scaling_factors = np.nanmean(np.nanmax(data[:, reference_columns], axis=0) / (data[:, reference_columns]), axis=1)

            nan_rows = np.where(np.isnan(scaling_factors))[0]
            if nan_rows.size > 0:
                print(f"{format_log_prefix('warn')} Rows {list(nan_rows)} have all missing reference values.")
                print(f"{format_log_prefix('info')} Falling back to row median normalization for these rows.")

                fallback = np.nanmedian(data[nan_rows, :], axis=1)
                fallback[fallback == 0] = np.nan  # avoid division by 0
                fallback_scale = np.nanmax(fallback) / fallback
                fallback_scale = np.where(np.isnan(fallback_scale), 1.0, fallback_scale)  # default to 1.0 if all else fails

                scaling_factors[nan_rows] = fallback_scale

            scaling_factors = np.where(np.isnan(scaling_factors), np.nanmean(scaling_factors), scaling_factors)
            data_norm = data * scaling_factors[:, None]

        elif method == 'robust_scale':
            # norm by robust_scale: Center to the median and component wise scale according to the interquartile range. See sklearn.preprocessing.robust_scale for more information.
            from sklearn.preprocessing import robust_scale
            data_norm = robust_scale(data, axis=1)

        elif method == 'quantile_transform':
            # norm by quantile_transform: Transform features using quantiles information. See sklearn.preprocessing.quantile_transform for more information.
            from sklearn.preprocessing import quantile_transform
            data_norm = quantile_transform(data, axis=1)

        else:
            raise ValueError(f"Unknown method: {method}")

        return data_norm
    
    def clean_X(self, on='prot', inplace=True, set_to=0, layer=None, to_sparse=False, backup_layer="X_preclean", verbose=True):
        """
        Replace NaNs in .prot.X or a specified .prot layer with a given value (default: 0).
        Optionally saves a backup of the original data to a layer (default: 'X_preclean').

        Parameters:
        - on (str): The attribute to clean ('prot' or 'obs').
        - inplace (bool): If True, overwrite .X or .layers[layer]. If False, return cleaned matrix.
        - set_to (float): Value to replace NaNs with.
        - layer (str or None): Target .prot layer to clean (default: .X).
        - to_sparse (bool): If True, convert result to sparse.
        - backup_layer (str or None): If inplace=True and layer=None, save .X to this layer (default: 'X_preclean').
        - verbose (bool): Print status messages.

        Returns:
        - np.ndarray or None: Cleaned matrix if inplace=False, otherwise None.
        """
        if not self._check_data(on):
            return
        adata = self.prot if on == 'prot' else self.pep

        # Choose source matrix
        X = adata.layers[layer] if layer else adata.X
        is_sparse = sparse.issparse(X)

        # Copy for manipulation
        X_clean = X.copy()
        nan_count = 0

        if is_sparse:
            nan_mask = np.isnan(X_clean.data)
            nan_count = np.sum(nan_mask)
            if nan_count > 0:
                X_clean.data[nan_mask] = set_to
        else:
            nan_mask = np.isnan(X_clean)
            nan_count = np.sum(nan_mask)
            X_clean[nan_mask] = set_to

        if to_sparse and not is_sparse:
            X_clean = sparse.csr_matrix(X_clean)

        # Apply result
        if inplace:
            if layer:
                self.prot.layers[layer] = X_clean
            else:
                # Save original .X if requested and not already backed up
                if backup_layer and backup_layer not in self.prot.layers:
                    self.prot.layers[backup_layer] = self.prot.X.copy()
                    if verbose:
                        print(f"{format_log_prefix('info')} Backed up .X to .layers['{backup_layer}']")
                self.prot.X = X_clean
            if verbose:
                print(f"{format_log_prefix('result')} Cleaned {'layer ' + layer if layer else '.X'}: replaced {nan_count} NaNs with {set_to}.")
        else:
            if verbose:
                print(f"{format_log_prefix('result')} Returning cleaned matrix: {nan_count} NaNs replaced with {set_to}.")
            return X_clean


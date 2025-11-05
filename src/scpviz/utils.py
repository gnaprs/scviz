"""
Utility functions for scpviz.

This module provides a collection of helper and processing functions used
throughout the scpviz package. They fall into four main categories:

## Text Utility Functions

Functions:
    format_log_prefix: Return standardized log prefixes for messages.
    format_class_filter: Standardize class/value inputs for filtering.

## Data Processing Functions

Functions:
    get_samplenames: Resolve sample names for given classes from `.obs`.
    get_classlist: Return unique class values for specified `.obs` columns.
    get_adata_layer: Safely extract data from `.X` or `.layers`.
    get_adata: Retrieve `.prot` or `.pep` AnnData from pAnnData.
    get_abundance: Wrapper to extract abundance data from pAnnData or AnnData.
    resolve_accessions: Map gene names or accessions to `.var_names`.
    get_upset_contents: Build contents for UpSet plots from pAnnData.
    get_upset_query: Query features present/absent in UpSet contents.
    filter: Legacy sample filtering (use `.filter_sample_values` instead).
    resolve_class_filter: Resolve class/value pairs and apply filtering.
    get_pep_prot_mapping: Determine peptide-to-protein mapping column.

## API Functions

Functions:
    get_uniprot_fields_worker: Low-level UniProt REST API query function (batch up to 1024).
    get_uniprot_fields: High-level UniProt API wrapper with batching.

## Statistical Test Functions

Functions:
    pairwise_log2fc: Compute pairwise median log2 fold change between groups.
    get_pca_importance: Identify most important features for PCA components.
    get_protein_clusters: Retrieve hierarchical clusters from stored linkage.

!!! warning
    Many of the functions in this module are **internal helpers** and not
    intended for direct end-user use. For filtering and abundance queries,
    prefer the corresponding `pAnnData` methods.

Example:
    To use this module, import it and call functions from your code as follows:
        ```python
        from scpviz import utils as scutils
        ```
    
Todo:
    * Add corrections for differential expression.
    * Add more examples for each function.
"""

from typing import List, Optional, Dict, Any, Union
from decimal import Decimal
from operator import index
from os import access
import re
import io
import requests
import warnings
import time

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, chi2_contingency, fisher_exact
from scipy.sparse import csr_matrix
from sklearn.impute import SimpleImputer, KNNImputer

import upsetplot
import anndata as ad
from scpviz import pAnnData

# Thoughts: functions that act on panndata and return only panndata should be panndata methods, utility functions should be in utils

# ----------------
# BASIC UTILITY FUNCTIONS

def format_log_prefix(level: str, indent=None) -> str:
    """
    Return a standardized log prefix with emoji and label.

    This helper formats log message prefixes consistently across scpviz,
    with optional indentation for nested output. Used internally for
    user-facing messages, warnings, errors, and updates.

    Args:
        level (str): Logging level keyword. Must be one of:

            - `"user"`: 🧭 [USER]  
            - `"search"`: 🔍 [SEARCH]  
            - `"info"`: ℹ️ [INFO]  
            - `"result"`: ✅ [OK]  
            - `"warn"`: ⚠️ [WARN]  
            - `"error"`: ❌ [ERROR]  
            - `"info_only"`: ℹ️  
            - `"filter_conditions"`: 🔸 (indented)  
            - `"result_only"`: ✅  
            - `"blank"`: empty string  
            - `"update"`: 🔄 [UPDATE]  
            - `"api"`: 🌐 [API]
            - `"update_only"`: 🔄  
            - `"warn_only"`: ⚠️
            - `"user_only"`: 🧭

        indent (int or None, optional): Indentation level override. Options:

            - `1`: no indent  
            - `2`: 5 spaces  
            - `3`: 10 spaces  

            If None, uses built-in default spacing (applied to most levels).

    Returns:
        str (str): A formatted log prefix with emoji and label.

    Raises:
        ValueError: If an unknown `level` string is provided.

    Example:
        Format an info prefix with default spacing:
        ```python
        from scpviz.utils import format_log_prefix
        format_log_prefix("info")
        ```

        ```
            ℹ️ [INFO]
        ```

        Format a warning prefix with explicit indent:
        ```python
        format_log_prefix("warn", indent=3)
        ```

        ```
                ⚠️ [WARN]
        ```
    """
    level = level.lower()
    base_prefixes = {
        "user": "🧭 [USER]",
        "search": "🔍 [SEARCH]",
        "info": "ℹ️ [INFO]",
        "result": "✅ [OK]",
        "warn": "⚠️ [WARN]",
        "error": "❌ [ERROR]",
        "info_only": "ℹ️",
        "filter_conditions": "     🔸 ",
        "result_only": "✅",
        "blank": "",
        "update": "🔄 [UPDATE]",
        "api": "🌐 [API]",
        "update_only": "🔄",
        "warn_only": "⚠️",
        "user_only": "🧭"
    }

    if level not in base_prefixes:
        raise ValueError(f"Unknown log level: {level}")

    prefix = base_prefixes[level]

    if indent is None:
        # Use default built-in spacing for all except info_only
        if level in ["info", "search", "result", "warn", "error"]:
            return "     " + prefix
        else:
            return prefix  # Default case, no indent (e.g. info_only)
    else:
        # Explicit indent override
        indent_spaces = {1: 0, 2: 5, 3: 10}
        space = " " * indent_spaces.get(indent, 0)
        return f"{space}{prefix}"

# ----------------
# DATA PROCESSING FUNCTIONS
# NOTE: get_samplenames and get_classlist are very similar, may want to explain better the difference (classlist is basically samplenames.unique?)
def get_samplenames(adata, classes):
    """
    Retrieve sample names for specified class values.

    This function resolves `.obs` metadata into sample-level identifiers
    (one name per row). It is typically used for plotting functions where
    sample names are required for labeling or grouping.

    Args:
        adata (anndata.AnnData): AnnData object containing sample metadata.

        classes (str or list of str): Column(s) in `.obs` used to build sample names.

            - str: return vlaues from a single column.
            - list of str: combine multiple columns per row with `", "`.

    Returns:
        sample_names (list of str): Sample names dervied from `.obs`.

    Example:
        Get sample names from a single metadata column:
            ```python
            samples = get_samplenames(adata, "cell_type")
            ```

        Combine multiple columns into sample identifiers:
            ```python
            samples = get_samplenames(adata, ["cell_type", "treatment"])
            ```

    Related Functions:
        get_classlist: Return unique class values (not per-sample names).
    """
    if classes is None:
        return None
    elif isinstance(classes, str):
        return adata.obs[classes].values.tolist()
    elif isinstance(classes, list):
        return adata.obs[classes].apply(lambda row: ', '.join(row.values.astype(str)), axis=1).values.tolist()
    else:
        raise ValueError("Invalid input for 'classes'. It should be None, a string, or a list of strings.")
    
def get_classlist(adata, classes = None, order = None):
    """
    Retrieve unique class values for specified metadata columns. Useful 
    for plot legends.

    Unlike `get_samplenames`, which returns one identifier per row/sample,
    this function extracts the set of unique class values for grouping
    purposes (e.g., plotting categories). Supports optional reordering.

    Args:
        adata (anndata.AnnData): AnnData object containing sample metadata.

        classes (str or list of str, optional): Column(s) in `.obs` to use.
            
            - None: combine all metadata columns up to the first `_quant` column.  
            - str: return unique values from one column.  
            - list of str: return unique combined values across multiple columns.  

        order (list of str, optional): Custom order of categories. Must exactly
            match the unique values; otherwise, a `ValueError` is raised.

    Returns:
        class_list (list of str): Unique class values in `.obs`, optionally reordered.

    Raises:
        ValueError: If invalid columns are provided, or if `order` does not
        match the unique class list.

    Example:
        Get unique values from one metadata column:
            ```python
            classes = get_classlist(adata, classes="cell_type")
            ```

        Combine two columns and return unique class labels:
            ```python
            classes = get_classlist(adata, classes=["cell_type", "treatment"])
            ```

        Reorder categories explicitly:
            ```python
            classes = get_classlist(
                adata, classes="cell_type", order=["A", "B", "C"]
                )
            ```

    Related Functions:
        get_samplenames: Return per-sample names (not unique class values).
    """

    if classes is None:
        # combine all .obs columns per row into one string
        # NOTE: might break, should use better method to filter out file-related columns
        quant_col_index = adata.obs.columns.get_loc(next(col for col in adata.obs.columns if "_quant" in col))
        selected_columns = adata.obs.iloc[:, :quant_col_index]
        classes_list = selected_columns.apply(lambda x: '_'.join(x), axis=1).unique()
        classes = selected_columns.columns.tolist()
    elif isinstance(classes, str):
        # check if classes is one of the columns of adata.obs
        if classes not in adata.obs.columns:
            raise ValueError(f"Invalid value for 'classes'. '{classes}' is not a column in adata.obs.")
        classes_list = adata.obs[classes].unique()
    elif isinstance(classes, list):
        # check if list has length 1
        if len(classes) == 1:
            classes_list = adata.obs[classes[0]].unique()
        # check if all classes are columns of adata.obs
        else:
            if not all([c in adata.obs.columns for c in classes]):
                raise ValueError(f"Invalid value for 'classes'. Not all elements in '{classes}' are columns in adata.obs.")
            classes_list = adata.obs[classes].apply(lambda x: '_'.join(x), axis=1).unique()
    else:
        raise ValueError("Invalid value for 'classes'. Must be None, a string or a list of strings.")

    if isinstance(classes_list, str):
        classes_list = [classes_list]
    if isinstance(order, str):
        order = [order]

    if order is not None:
        # check if order list matches classes_list
        missing_elements = set(classes_list) - set(order)
        extra_elements = set(order) - set(classes_list)
        # Print missing and extra elements if any
        if missing_elements or extra_elements:
            if missing_elements:
                print(f"Missing elements in 'order': {missing_elements}")
            if extra_elements:
                print(f"Extra elements in 'order': {extra_elements}")
            raise ValueError("The 'order' list does not match 'classes_list'.")
        # if they match, then reorder classes_list to match order
        classes_list = order

    return classes_list

def get_adata_layer(adata, layer):
    """
    Safely extract layer data as dense numpy array.

    This helper returns the requested layer as a dense `numpy.ndarray`,
    ensuring compatibility for downstream operations. Supports both
    `.X` and `.layers[...]`.

    Args:
        adata (anndata.AnnData): AnnData object containing data matrices.

        layer (str): Layer key.  
            - `"X"`: return the main data matrix.  
            - any other str: return the corresponding entry from `.layers`. E.g. "X_norm"

    Returns:
        data (numpy.ndarray): Dense matrix representation of the requested layer.
    """
    if layer == "X":
        data = adata.X
    elif layer in adata.layers:
        data = adata.layers[layer]
    else:
        raise ValueError(f"Layer '{layer}' not found in .layers and is not 'X'.")

    return data.toarray() if hasattr(data, 'toarray') else data

def get_adata(pdata, on = 'protein'):
    """
    Retrieve the protein- or peptide-level AnnData object from a pAnnData container.

    Args:
        pdata (pAnnData): The parent pAnnData object containing both protein- and peptide-level data.

        on (str): Which data object to return.  
            - `"protein"`: return `pdata.prot`  
            - `"peptide"`: return `pdata.pep`  

    Returns:
        adata (anndata.AnnData): The requested AnnData object.
    """

    if on in ('protein','prot'):
        return pdata.prot
    elif on in ('peptide','pep'):
        return pdata.pep
    else:
        raise ValueError("Invalid value for 'on'. Options are 'protein' or 'peptide'.")

def get_abundance(pdata, *args, **kwargs):
    """
    Wrapper to extract abundance from either pAnnData or AnnData.

    This is a convenience wrapper that dispatches to the appropriate method:
    - If `pdata` is a `pAnnData` object, it calls `pdata.get_abundance()`.
    - If `pdata` is an `AnnData` object, it falls back to the internal
      helper `_get_abundance_from_adata`.

    Args:
        pdata (pAnnData or anndata.AnnData): Input object to extract abundance from.
        *args: Positional arguments forwarded to `get_abundance`.
        **kwargs: Keyword arguments forwarded to `get_abundance`.

    Note:
        See `pAnnData.get_abundance` for full parameter documentation. Briefly,

            - namelist (list of str, optional): List of accessions or gene names to extract.
            - layer (str): Data layer name (default = "X").
            - on (str): "protein" or "peptide".
            - classes (str or list of str, optional): Sample-level `.obs` column(s) to include.
            - log (bool): If True, applies log2 transform to abundance values.
            - x_label (str): Label features by "gene" or "accession".

    Returns:
        df (pandas.DataFrame): Long-form abundance DataFrame, optionally with
        sample metadata and protein/peptide annotations.

    See Also:
        - :func:`pAnnData.get_abundance` (EditingMixin): Full-featured version with detailed docs.
        - get_adata_layer: Helper to access abundance matrices from AnnData layers.
    """
    if hasattr(pdata, "get_abundance"):
        return pdata.get_abundance(*args, **kwargs)
    else:
        # Fall back to internal logic (can move to a private function if cleaner)
        return _get_abundance_from_adata(pdata, *args, **kwargs)

def _get_abundance_from_adata(adata, namelist=None, layer='X', log=True,
                               x_label='gene', classes=None, gene_col="Genes"):
    """
    Abundance extraction for plain AnnData, including gene/accession support.
    """

    # Resolve gene names → accessions
    if namelist:
        resolved = resolve_accessions(adata, namelist, gene_col=gene_col)
        adata = adata[:, resolved]

    # Extract matrix
    X = adata.layers[layer] if layer in adata.layers else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    df = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names).reset_index()
    df = df.melt(id_vars="index", var_name="accession", value_name="abundance")
    df = df.rename(columns={"index": "cell"})

    df = df.merge(adata.obs.reset_index(), left_on="cell", right_on="index")

    gene_map = adata.var["Genes"].to_dict() if "Genes" in adata.var else {}
    df['gene'] = df['accession'].map(gene_map)
    df['x_label_name'] = df['gene'].fillna(df['accession']) if x_label == 'gene' else df['accession']

    if classes:
        df['class'] = df[classes] if isinstance(classes, str) else df[classes].astype(str).agg('_'.join, axis=1)
    else:
        df['class'] = 'all'

    if log:
        df['log2_abundance'] = np.log2(np.clip(df['abundance'], 1e-6, None))

    return df

def resolve_accessions(adata, namelist, gene_col="Genes", gene_map=None):
    """
    Resolve gene or accession names to accession IDs from `.var_names`.

    This function maps user-specified identifiers (gene names or accession IDs)
    to the canonical accession IDs in an AnnData or pAnnData object. It first
    checks `.var_names` for exact matches, then optionally resolves gene names
    via a specified column (default `"Genes"`). Unmatched names are reported.

    Args:
        adata (AnnData or pAnnData): AnnData-like object containing `.var`.
        namelist (list of str): Input identifiers to resolve (genes or accessions).
        gene_col (str): Column in `.var` containing gene names (default: `"Genes"`).
        gene_map (dict, optional): Precomputed mapping of gene → accession. If None,
            a mapping is constructed from `gene_col`.

    Returns:
        resolved (list of str): List of accession IDs corresponding to the input names.

    Raises:
        ValueError: If none of the provided names can be resolved to `.var_names`
            or the gene column.

    Example:
        Resolve gene symbols to accession IDs:
            ```python
            accs = resolve_accessions(adata, namelist=["UBE4B", "GAPDH"])
            ```

        Resolve accessions directly:    
            ```python
            accs = resolve_accessions(adata, namelist=["P12345", "Q67890"])
            ```
    
    Related Functions:
        - get_gene_maps: Build full accession → gene mapping dictionaries.
        - get_abundance: Extract abundance values by gene or accession.
    """
    import pandas as pd

    if not namelist:
        return None

    var_names = adata.var_names.astype(str)

    # Use passed-in gene_map or build one
    if gene_map is None:
        gene_map = {}
        if gene_col in adata.var.columns:
            for acc, gene in zip(var_names, adata.var[gene_col]):
                if pd.notna(gene):
                    gene_map[str(gene)] = acc

    resolved, unmatched = [], []
    for name in namelist:
        name = str(name)
        if name in var_names:
            resolved.append(name)
        elif name in gene_map:
            resolved.append(gene_map[name])
        else:
            unmatched.append(name)

    if not resolved:
        raise ValueError(
            f"No valid names found in `namelist`: {namelist}.\n"
            f"Check against .var_names or '{gene_col}' column."
        )

    if unmatched:
        print(f"{format_log_prefix('warn')}resolve_accessions: Unmatched names:")
        for u in unmatched:
            print(f"  - {u}")

    return resolved

def get_upset_contents(pdata, classes, on = 'protein', upsetForm = True, debug=False):
    """
    Construct contents for an UpSet plot from a pAnnData object.

    This function extracts feature sets (proteins or peptides) present in
    specified sample classes and returns them either as a dictionary or
    in an `upsetplot`-compatible format.

    Args:
        pdata (pAnnData): The pAnnData object containing `.prot` and `.pep`.
        classes (str or list of str): Metadata column(s) in `.obs` to define sample groups.
            Examples: `"cell_type"`, or `["cell_type", "treatment"]`.
        on (str): Data level to use. Options are `"protein"` (default) or `"peptide"`.
        upsetForm (bool): If True, return an `UpSet`-compatible DataFrame via
            `upsetplot.from_contents`. If False, return a raw dictionary.
        debug (bool): If True, print filtering steps and class resolution details.

    Returns:
        upset_data (pandas.DataFrame): Binary presence/absence DataFrame for use with
            `upsetplot.UpSet`, if `upsetForm=True`.
        upset_dict (dict): Mapping of class → list of present features,
            if `upsetForm=False`.

    Raises:
        ValueError: If `on` is not `"protein"` or `"peptide"`.

    Example:
        Get contents for an UpSet plot of sample classes:
            ```python
            upset_data = get_upset_contents(pdata, classes="treatment")
            from upsetplot import UpSet
            UpSet(upset_data, subset_size="count").plot()
            ```

        Retrieve raw dictionary of sets instead:
            ```python
            upset_dict = get_upset_contents(pdata, classes="treatment", upsetForm=False)
            ```

        Query proteins from a set and highlight them in a plot:
            ```python
            upset_data = scutils.get_upset_contents(pdata, classes="condition")
            prot_df = scutils.get_upset_query(upset_data, present=["treated"], absent=["control"])
            scplt.plot_rankquant(ax, pdata, classes="condition", cmap=cmaps, color=colors)
            scplt.mark_rankquant(ax, pdata, mark_df=prot_df, class_values=["treated"], color="black")
            ```

    Related Functions:
        - plot_upset: Plot UpSet diagrams directly.
        - plot_venn: Plot Venn diagrams for up to 3 sets.
    """

    if on == 'protein':
        adata = pdata.prot
    elif on == 'peptide':
        adata = pdata.pep
    else:
        raise ValueError("Invalid value for 'on'. Options are 'protein' or 'peptide'.")

    # Common error: if classes is a list with only one element, unpack it
    if isinstance(classes, list) and len(classes) == 1:
        classes = classes[0]

    classes_list = get_classlist(adata, classes)
    upset_dict = {}

    for j, class_value in enumerate(classes_list):
        data_filtered = resolve_class_filter(adata, classes, class_value, debug=True)

        # get proteins that are present in the filtered data (at least one value is not NaN, not 0)
        X = data_filtered.X.toarray()
        mask_present = (~np.isnan(X)) & (X != 0)
        prot_present = data_filtered.var_names[mask_present.sum(axis=0) > 0]
        upset_dict[class_value] = prot_present.tolist()

    if upsetForm:
        upset_data = upsetplot.from_contents(upset_dict)
        return upset_data
    
    else:
        return upset_dict

def get_upset_query(upset_content, present, absent):
    """
    Query features from UpSet contents given inclusion and exclusion criteria.

    This function extracts the set of features (proteins or peptides) that are
    present in all specified groups and absent in others. It then queries
    UniProt metadata for the resulting accessions.

    Args:
        upset_content (pandas.DataFrame): Output from `get_upset_contents` with
            presence/absence encoding of features.
        present (list of str): List of groups in which the features must be present.
        absent (list of str): List of groups in which the features must be absent.

    Returns:
        prot_query_df (pandas.DataFrame): DataFrame of features matching the query,
        annotated with UniProt metadata via `get_uniprot_fields`.

    Example:
        Query proteins unique to one group and highlight them in a plot:
            ```python
            upset_data = scutils.get_upset_contents(pdata, classes="condition")
            prot_df = scutils.get_upset_query(upset_data, present=["treated"], absent=["control"])
            scplt.plot_rankquant(ax, pdata, classes="condition", cmap=cmaps, color=colors)
            scplt.mark_rankquant(ax, pdata, mark_df=prot_df, class_values=["treated"], color="black")
            ```

    Related Functions:
        - get_upset_contents: Generate presence/absence sets for UpSet analysis.
        - plot_upset: Plot UpSet diagrams from class-based sets.
    """
    prot_query = upsetplot.query(upset_content, present=present, absent=absent).data['id'].tolist()
    prot_query_df = get_uniprot_fields(prot_query,verbose=False)

    return prot_query_df

# TODO: LEGACY, CHECK TO MAKE SURE IT"S NOT BEING USED ANYWHERE ELSE
def filter(pdata, class_type, values, exact_cases = False, debug = False):
    """
    Legacy-style filtering of samples in pAnnData or AnnData objects.

    This function filters samples based on metadata values using the older
    `(class_type, values)` interface. For pAnnData objects, it automatically
    delegates to `.filter_sample_values()` after converting the input into the
    recommended dictionary-style format.

    !!! warning

        For pAnnData users, prefer `.filter_sample_values()` with dictionary-style
        input, as it is more flexible and consistent. The `filter()` utility is
        retained primarily for backward compatibility and direct AnnData usage.


    Args:
        pdata (pAnnData or AnnData): Input data object to filter.
        class_type (str or list of str): Metadata field(s) in `.obs` to filter on.
            Example: `"treatment"`, or `["cell_type", "treatment"]`.
        values (list, dict, or list of dict): Metadata values to match.
            - If `exact_cases=False`: Provide a dictionary or list-of-values per class.
            - If `exact_cases=True`: Provide a list of dictionaries specifying
              exact combinations across fields.
        exact_cases (bool): Whether to interpret `values` as exact combinations (AND logic).
            Defaults to False, which applies OR logic within each class type.
        debug (bool): If True, print the query string used for filtering.

    Returns:
        filtered (pAnnData or AnnData): A filtered object of the same type as `pdata`.


    Raises:
        ValueError: If input types are invalid, if fields are missing in `.obs`,
            or if `values` format does not match `exact_cases`.

    Example:
        Filter samples by a single metadata field:
            ```python
            samples = utils.filter(pdata, class_type="treatment", values="kd")
            ```

        Filter by multiple fields with OR logic: 
            ```python
            samples = utils.filter(
                    adata,
                    class_type=["cell_type", "treatment"],
                    values=[["wt", "kd"], ["control", "treatment"]]
                ) 
            # returns samples where cell_type is either 'wt' or 'kd' and treatment is either 'control' or 'treatment'
            ```

        Filter by exact case combinations:
            ```python 
            samples = utils.filter(
                    adata,
                    class_type=["cell_type", "treatment"],
                    values=[{"cell_type": "wt", "treatment": "control"},
                            {"cell_type": "kd", "treatment": "treatment"}],
                    exact_cases=True
                )
            # returns samples where cell_type is 'wt' and treatment is 'kd', or cell_type is 'control' and treatment is 'treatment'
            ```
    """
    
    if hasattr(pdata, "filter_sample_values"):
        warnings.warn(
            "You passed a pAnnData object to `filter()`. "
            "It is recommended to use `pdata.filter_sample_values()` directly.",
            UserWarning)
        
        print("UserWarning: It is recommended to use the class method `.filter_sample_values()` with dictionary-style input for cleaner and more consistent filtering.")

    formatted_values = format_class_filter(class_type, values, exact_cases)
    
    # pAnnData input
    if hasattr(pdata, "filter_sample_values"):
        return pdata.filter_sample_values(
            values=formatted_values,
            exact_cases=exact_cases,
            debug=debug,
            return_copy=True
        )

    # plain AnnData input
    elif isinstance(pdata, ad.AnnData):
        adata = pdata
        obs_keys = adata.obs.columns

        if exact_cases:
            if not isinstance(formatted_values, list) or not all(isinstance(v, dict) for v in formatted_values):
                raise ValueError("When exact_cases=True, `values` must be a list of dictionaries.")

            for case in formatted_values:
                if not case:
                    raise ValueError("Empty dictionary found in values.")
                for key in case:
                    if key not in obs_keys:
                        raise ValueError(f"Field '{key}' not found in adata.obs.")

            query = " | ".join([
                " & ".join([
                    f"(adata.obs['{k}'] == '{v}')" for k, v in case.items()
                ])
                for case in formatted_values
            ])

        else:
            if not isinstance(formatted_values, dict):
                raise ValueError("When exact_cases=False, `values` must be a dictionary.")

            for key in formatted_values:
                if key not in obs_keys:
                    raise ValueError(f"Field '{key}' not found in adata.obs.")

            query_parts = []
            for k, v in formatted_values.items():
                v_list = v if isinstance(v, list) else [v]
                part = " | ".join([f"(adata.obs['{k}'] == '{val}')" for val in v_list])
                query_parts.append(f"({part})")
            query = " & ".join(query_parts)

        if debug:
            print(f"Filter query: {query}")

        return adata[eval(query)]

    else:
        raise ValueError("Input must be a pAnnData or AnnData object.")

def format_class_filter(classes, class_value, exact_cases=False):
    """
    Convert legacy-style filter input into dictionary-style format.

    This function standardizes `(classes, class_value)` input into the dictionary
    format expected by `pAnnData.filter_sample_values()`. It supports both loose
    OR-style filtering and exact case matching across multiple metadata fields.

    Args:
        classes (str or list of str): Metadata field(s) to filter on.
            Example: `"treatment"` or `["cellline", "treatment"]`.
        class_value (str, list of str, or list of list): Values to filter by.
            - str: May be underscore-joined (e.g. `"kd_AS"`).
            - list of str: Multiple values, interpreted as OR (if `exact_cases=False`)
              or split into combinations (if `exact_cases=True`).
            - list of list: Each inner list defines a full set of values across classes.
        exact_cases (bool): If True, return a list of dictionaries representing
            exact combinations across fields. If False, return a dictionary with
            OR logic applied.

    Returns:
        formatted (dict or list of dict): Dictionary-style filter input compatible
        with `.filter_sample_values()`.

    Raises:
        ValueError: If input shapes are inconsistent with the number of classes,
            or if `class_value` entries are not valid strings/lists.

    Example:
        Single class with OR logic:
            ```python
            format_class_filter("treatment", ["kd", "sc"])
            ```
            ```
            {'treatment': ['kd', 'sc']}
            ```

        Multiple classes with loose matching:
            ```python
            format_class_filter(["cellline", "treatment"], ["AS", "kd"])
            ```
            ```
            {'cellline': 'AS', 'treatment': 'kd'}
            ```

        Multiple classes with exact cases (underscore-joined strings):
            ```python
            format_class_filter(
                ["cellline", "treatment"],
                ["AS_kd", "BE_sc"],
                exact_cases=True
            )
            ```
            ```
            [{'cellline': 'AS', 'treatment': 'kd'},
             {'cellline': 'BE', 'treatment': 'sc'}]
            ```

        Multiple classes with exact cases (list of lists):
            ```python 
            format_class_filter(
                ["cellline", "treatment"],
                [["AS", "kd"], ["BE", "sc"]],
                exact_cases=True
            )
            ```
            ```
            # [{'cellline': 'AS', 'treatment': 'kd'},
             {'cellline': 'BE', 'treatment': 'sc'}]
            ```

    !!! warning "Note"

        This function is primarily used internally by `utils.filter()` and
        `pAnnData.filter_sample_values()`. End users should generally call
        `.filter_sample_values()` directly on `pAnnData` objects instead of
        using this helper.
    """

    if isinstance(classes, str):
        # Simple case: one class
        if isinstance(class_value, list) and exact_cases:
            return [{classes: val} for val in class_value]
        else:
            return {classes: class_value}

    elif isinstance(classes, list):
        if exact_cases:
            if isinstance(class_value, str):
                class_value = [class_value]

            formatted = []
            for entry in class_value:
                if isinstance(entry, str):
                    values = entry.split('_')
                elif isinstance(entry, list):
                    values = entry
                else:
                    raise ValueError("Each class_value entry must be a string or a list.")

                if len(values) != len(classes):
                    raise ValueError("Each class_value entry must match the number of classes.")
                formatted.append({cls: val for cls, val in zip(classes, values)})

            return formatted

        else:
            # loose match — OR within each class
            if isinstance(class_value, str):
                values = class_value.split('_')
            else:
                values = class_value
            if len(values) != len(classes):
                raise ValueError("class_value must align with the number of classes.")
            return {cls: val for cls, val in zip(classes, values)}

    else:
        raise ValueError("Invalid input: `classes` should be a string or list of strings.")

def resolve_class_filter(adata, classes, class_value, debug=False, *, filter_func=None):
    """
    Resolve `(classes, class_value)` inputs and apply filtering.

    This helper standardizes class/value pairs into dictionary-style filters
    and applies them to an AnnData or pAnnData object. It is primarily used
    internally by plotting and analysis functions.

    Args:
        adata (AnnData or pAnnData): Input data object to filter.
        classes (str or list of str): Metadata field(s) used for filtering.
        class_value (str or list of str): Values corresponding to `classes`.
        debug (bool): If True, print resolved class/value pairs.
        filter_func (callable, optional): Filtering function to apply.
            Defaults to `utils.filter`.

    Returns:
        filtered (AnnData or pAnnData): Subset of the input object, same type as `adata`.

    !!! warning
        This is an internal helper for use inside functions such as
        `plot_rankquant` and `plot_raincloud`. End users should call
        `pAnnData.filter_sample_values()` instead.

    Related Functions:
        - filter: Legacy utility for sample filtering.
        - format_class_filter: Standardizes filter inputs.
        - pAnnData.filter_sample_values: Recommended user-facing filter method.
    """

    from scpviz import utils  # safe to do here in case of circular import

    if isinstance(classes, str):
        values = class_value
    else:
        values = class_value.split('_')

    if debug:
        print(f"Classes: {classes}, Values: {values}")

    if filter_func is None:
        filter_func = utils.filter

    return filter_func(adata, classes, values, debug=debug)

def get_pep_prot_mapping(pdata, return_series=False):
    """
    Retrieve the peptide-to-protein mapping column or mapping values.

    This function resolves the appropriate `.pep.var` column for peptide-to-protein
    mapping based on the data source recorded in `pdata.metadata["source"]`.

    Args:
        pdata (pAnnData): The annotated proteomics object containing `.metadata` and `.pep`.
        return_series (bool): If True, return a pandas Series of peptide-to-protein
            mappings. If False (default), return the column name as a string.

    Returns:
        col (str): Column name in `.pep.var` containing peptide-to-protein mapping,
        if `return_series=False`.
        mapping (pandas.Series): Series mapping peptides to proteins,
        if `return_series=True`.

    Raises:
        ValueError: If the data source is unrecognized or no valid mapping column is found.

    Note:
        The mapping column depends on the import source:
        
        - Proteome Discoverer → `"Master Protein Accessions"`
        - DIA-NN → `"Protein.Group"`
        - MaxQuant → `"Leading razor protein"`
    """
    source = pdata.metadata.get("source", "").lower()

    if source == "proteomediscoverer":
        col = "Master Protein Accessions"
    elif source == "diann":
        col = "Protein.Group"
    elif source == "maxquant":
        col = "Leading razor protein"
    else:
        raise ValueError(f"Unknown data source '{source}' — cannot determine peptide-to-protein mapping.")

    if return_series:
        return pdata.pep.var[col]

    return col

# ----------------
# API FUNCTIONS (STRING functions in enrichment.py)
def get_uniprot_fields_worker(prot_list, search_fields=None, verbose = False):
    """
    Query UniProt for a batch of protein accessions.

    This function sends requests to the UniProt REST API for up to 1024 proteins
    at a time and returns the requested fields as a DataFrame. It handles isoform
    accessions, fallback queries, and UniProt ID redirects automatically.

    Args:
        prot_list (list of str): List of protein accessions or IDs.
        search_fields (list of str): UniProt return fields.
            See: https://www.uniprot.org/help/return_fields
        verbose (bool): If True, print progress messages and missing accessions.

    Returns:
        df (pandas.DataFrame): DataFrame containing UniProt metadata for the input proteins.

    Raises:
        ValueError: If `query_type` is unknown or the data source cannot be resolved.

    !!! info
        - This function is intended as a **worker** and is usually called by
          `get_uniprot_fields`.
        - It automatically resolves canonical vs. isoform accessions and will
          attempt UniProt ID mapping if some accessions cannot be found.

    Related Functions:
        - get_uniprot_fields: High-level batch UniProt query wrapper.
    """

    base_url = 'https://rest.uniprot.org/uniprotkb/stream'
    fields = "%2C".join(search_fields)
    format_type = 'tsv'
    
    def query_uniprot_batch(ids, query_type="accession"):
        if not ids:
            return pd.DataFrame()

        if query_type == "accession":
            query_parts = [f"%28accession%3A{id}%29" for id in ids]
        elif query_type == "id":
            query_parts = [f"%28id%3A{id}%29" for id in ids]
        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        query = "+OR+".join(query_parts)
        full_query = f"%28{query}%29"
        url = f'{base_url}?fields={fields}&format={format_type}&query={full_query}'

        if verbose:
            print(f"Querying UniProt ({query_type}, TSV mode) for {len(ids)} proteins")

        results = requests.get(url)
        results.raise_for_status()

        # Handle empty response gracefully
        if not results.text.strip():
            print(f"{format_log_prefix('warn_only', 2)} UniProt returned empty response for {len(ids)} proteins.")
            return pd.DataFrame()

        return pd.read_csv(io.StringIO(results.text), sep="\t")

    if verbose:
        print(f"{format_log_prefix('API', 1)} Querying UniProt for {len(prot_list)} total proteins [TSV mode].")
    
    def resolve_uniprot_redirects(accessions, from_db='UniProtKB_AC-ID', to_db='UniProtKB'):
        url = 'https://rest.uniprot.org/idmapping/run'
        data = {'from': from_db, 'to': to_db, 'ids': ','.join(accessions)}

        res = requests.post(url, data=data)
        res.raise_for_status()
        job_id = res.json()['jobId']

        # Poll until job is complete
        while True:
            status = requests.get(f"https://rest.uniprot.org/idmapping/status/{job_id}").json()
            if status.get("jobStatus") == "RUNNING":
                time.sleep(1)
            else:
                break

        # Get results
        results = requests.get(f"https://rest.uniprot.org/idmapping/uniprotkb/results/{job_id}").json()
        mapping = {item['from']: item['to']['primaryAccession'] for item in results.get('results', [])}
        return mapping

    # Split isoform vs canonical accessions
    isoform_ids = [acc for acc in prot_list if '-' in acc]
    canonical_ids = [acc for acc in prot_list if '-' not in acc]

    df_canonical = query_uniprot_batch(canonical_ids, query_type="accession")
    df_isoform = query_uniprot_batch(isoform_ids, query_type="accession")

    # Identify any isoforms that weren't found
    found_isoform_ids = set(df_isoform['Entry']) if not df_isoform.empty else set()
    missing_isoforms = [acc for acc in isoform_ids if acc not in found_isoform_ids]

    if missing_isoforms and verbose:
        print(f"{format_log_prefix('info_only', 3)} Attempting fallback query for {len(missing_isoforms)} isoform base IDs")

    # Attempt fallback query using base accessions
    fallback_ids = list(set([id.split('-')[0] for id in missing_isoforms]))
    df_fallback = query_uniprot_batch(fallback_ids, query_type="id")

    # Combine all DataFrames
    df = pd.concat([df_canonical, df_isoform, df_fallback], ignore_index=True)

    # Final pass: insert missing rows if still unresolved
    found_entries = set(df['Entry']) if 'Entry' in df.columns else set()
    still_missing = set(prot_list) - found_entries

    if still_missing:
        if verbose:
            print(f"{format_log_prefix('info_only', 3)} Attempting UniProt ID redirect for {len(still_missing)} unresolved accessions.")
        redirect_map = resolve_uniprot_redirects(list(still_missing))
        if redirect_map:
            redirected_ids = list(redirect_map.values())
            df_redirected = query_uniprot_batch(redirected_ids, query_type="accession")
            
            # Remap back to original accession
            inv_map = {v: k for k, v in redirect_map.items()}
            if 'Entry' in df_redirected.columns:
                df_redirected['Entry'] = df_redirected['Entry'].apply(lambda x: inv_map.get(x, x))

            df = pd.concat([df, df_redirected], ignore_index=True)

            resolved = set(redirect_map.keys())
            still_missing -= resolved

    # Step 5: Fill in placeholders for totally missing accessions
    if still_missing:
        print(f"{format_log_prefix('warn_only', 3)} Proteins not found in UniProt: {list(still_missing)[:5]}") if verbose else None
        missing_df = pd.DataFrame({'Entry': list(still_missing)})
        for col in search_fields:
            if col != 'accession' and col not in missing_df.columns:
                missing_df[col] = np.nan
        df = pd.concat([df, missing_df], ignore_index=True)
    
    if 'STRING' in df.columns:
        # keep first STRING ID (or join all if you prefer)
        df['xref_string'] = df['STRING'].apply(
            lambda s: str(s).split(';')[0].strip() if pd.notna(s) and str(s).strip() else np.nan
        )
        df.drop(columns=['STRING'], inplace=True)

    return df

def get_uniprot_fields(
    prot_list,
    search_fields=['accession', 'id', 'protein_name', 'gene_primary', 'gene_names',
                   'organism_id', 'go', 'go_f', 'go_c', 'go_p',
                   'cc_interaction', 'xref_string'],
    batch_size=100,
    verbose=True,
    standardize=True,
    worker_verbose=False,
):
    """
    Retrieve UniProt metadata for a list of protein accessions.

    This function wraps `get_uniprot_fields_worker` to handle batching of
    protein IDs, returning results as a single DataFrame.

    Args:
        prot_list (list of str): List of protein accessions.
        search_fields (list of str): UniProt fields to return.
            Defaults include accession, gene names, GO terms, and STRING IDs.
        batch_size (int): Number of accessions per batch (max 1024, default=100).
        verbose (bool): If True, print progress messages.
        standardize (bool): If True (default), normalize UniProt column names
            to canonical lowercase keys (e.g., "gene_primary", "organism_id",
            "xref_string") for consistent downstream processing.

    Returns:
        df (pandas.DataFrame): DataFrame containing UniProt metadata for the input proteins.

    Example:
        Query UniProt for a small set of proteins:
            ```python
            proteins = ["P40925", "P40926"]
            df = get_uniprot_fields(proteins)
            df[["Entry", "Gene Names", "Organism Id"]].head()
            ```

        Retrieve raw UniProt field names without renaming:
            >>> df_raw = get_uniprot_fields(proteins, standardize=False)

    Related Functions:
        - get_uniprot_fields_worker: Worker function that handles low-level UniProt API queries.
        - standardize_uniprot_columns: Helper used internally for column normalization.
    """

    # --- Ensure 'accession' field comes first (UniProt requirement)
    search_fields = ["accession"] + [f for f in search_fields if f != "accession"]

    # --- Split IDs into batches
    batches = [prot_list[i:i + batch_size] for i in range(0, len(prot_list), batch_size)]
    all_results = []

    for i, batch in enumerate(batches, start=1):
        if verbose:
            print(
                f"{format_log_prefix('api', indent=2)} Querying UniProt for batch {i}/{len(batches)} "
                f"({len(batch)} proteins) [fields: {', '.join(search_fields)}]"
            )

            if len(batches) > 1:
                print(f"{format_log_prefix('info_only', indent=3)} Processing batch {i}/{len(batches)}...")

        try:
            batch_df = get_uniprot_fields_worker(batch, search_fields, verbose=worker_verbose)
            if standardize:
                batch_df = standardize_uniprot_columns(batch_df)
            all_results.append(batch_df)
        except Exception as e:
            print(f"{format_log_prefix('warn')} Failed batch {i}: {e}")
            continue

    if not all_results:
        if verbose:
            print(f"{format_log_prefix('warn')} No results retrieved from UniProt.")
        return pd.DataFrame()

    full_method_df = pd.concat(all_results, ignore_index=True)
    if verbose:
        print(f"{format_log_prefix('result_only', 2)} Retrieved UniProt metadata for {len(full_method_df)} entries.")

    return full_method_df

def standardize_uniprot_columns(df):
    """
    Normalize UniProt DataFrame column names to a consistent lowercase, snake_case schema.

    This ensures stability across UniProt REST API version changes while keeping
    the user informed only when critical fields are affected.

    Args:
        df (pd.DataFrame): Raw UniProt metadata table.

    Returns:
        pd.DataFrame: Copy of the DataFrame with standardized column names.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        return df

    rename_map = {}
    aliases = {
        # identifiers
        "entry": "accession",
        "entry_name": "id",
        "accession": "accession",
        "primaryaccession": "accession",

        # gene fields
        "gene_names_primary": "gene_primary",
        "gene_name_primary": "gene_primary",
        "gene_primary_name": "gene_primary",
        "gene_primary": "gene_primary",
        "gene_primaryname": "gene_primary",
        "gene_primary_name_": "gene_primary",
        "gene_primaryname_": "gene_primary",

        # organism fields
        "organism_id": "organism_id",
        "organism_identifier": "organism_id",
        "organismid": "organism_id",

        # STRING / cross-reference
        "cross_reference_string": "xref_string",
        "xref_string_id": "xref_string",
        "crossreference_string": "xref_string",
        "string": "xref_string",
        "string_id": "xref_string",
        "xref_string": "xref_string",
    }

    # critical canonical fields we care about if changed or missing
    critical_fields = {"accession", "gene_primary", "organism_id", "xref_string"}

    # known benign patterns — don't warn if these change
    benign_patterns = {
        "gene_ontology",
        "go",
        "gene_names",      # non-primary gene list
        "protein_name",    # descriptive only
        "cc_interaction",  # crossref metadata
    }

    for col in df.columns:
        norm = (
            re.sub(r"[^a-z0-9]+", "_", col.lower())
            .strip("_")
            .replace("__", "_")
        )

        mapped = aliases.get(norm, None)

        if mapped:
            rename_map[col] = mapped
        else:
            # warn only if this looks like a drifted critical column
            if (
                any(k in norm for k in ["accession", "gene", "organism", "string"])
                and not any(p in norm for p in benign_patterns)
            ):
                warnings.warn(
                    f"[standardize_uniprot_columns] ⚠️ Unrecognized UniProt column '{col}' "
                    f"(normalized='{norm}') — may affect critical mapping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            rename_map[col] = norm  # keep normalized fallback name

    df = df.rename(columns=rename_map)
    # verify that all critical fields exist at least once
    missing_critical = [c for c in critical_fields if c not in df.columns]
    if missing_critical:
        warnings.warn(
            f"[standardize_uniprot_columns] Missing expected UniProt columns: {', '.join(missing_critical)}",
            RuntimeWarning,
            stacklevel=2,
        )

    return df.rename(columns=rename_map)

# ----------------
# STATISTICAL TEST FUNCTIONS
def pairwise_log2fc(data1, data2):
    """
    Compute pairwise median log2 fold change (log2FC) between two groups.

    This function calculates all pairwise log2 ratios between features in
    two groups of samples and returns the median value per feature. It is
    primarily used as a helper for fold-change strategies in `pAnnData.de()`.

    Args:
        data1 (numpy.ndarray): Array of shape `(n_samples_group1, n_features)`
            containing abundance values for group 1.
        data2 (numpy.ndarray): Array of shape `(n_samples_group2, n_features)`
            containing abundance values for group 2.

    Returns:
        median_log2fc (numpy.ndarray): Array of shape `(n_features,)` containing
        the median pairwise log2 fold change for each feature.

    Note:
        This is an internal helper for differential expression calculations.
        End users should call `pAnnData.de()` instead of using this function directly.

    Related Functions:
        - pAnnData.de: Differential expression analysis with multiple fold change strategies.
    """
    n1, n2 = data1.shape[0], data2.shape[0]

    # data1[:, None, :] has shape (n1, 1, n_features)
    # data2[None, :, :] has shape (1, n2, n_features)
    # The result is an array of shape (n1, n2, n_features)
    with np.errstate(divide='ignore', invalid='ignore'):
        pairwise_ratios = np.log2(data1[:, None, :] / data2[None, :, :])  # (n1, n2, features)
        pairwise_flat = pairwise_ratios.reshape(-1, data1.shape[1])

    # Identify columns that are entirely NaN
    mask_all_nan = np.all(np.isnan(pairwise_flat), axis=0)
    median_fc = np.full(data1.shape[1], np.nan, dtype=float)

    # Compute only on valid columns
    if not np.all(mask_all_nan):
        valid_cols = ~mask_all_nan
        median_fc[valid_cols] = np.nanmedian(pairwise_flat[:, valid_cols], axis=0)

    # # Reshape to (n1*n2, n_features) and compute the median along the first axis.
    # median_fc = np.nanmedian(pairwise_ratios.reshape(-1, data1.shape[1]), axis=0)
    return median_fc

def get_pca_importance(model: Union[dict, 'sklearn.decomposition.PCA'], initial_feature_names: List[str], n: int = 1) -> pd.DataFrame:
    """
    Identify the most important features for each principal component.

    This function ranks features by their absolute PCA loading values and
    extracts the top contributors for each principal component.

    Args:
        model (sklearn.decomposition.PCA or dict): Either a fitted PCA model
            from scikit-learn, or a dictionary with key `"PCs"`
            (array-like, shape: `(n_components, n_features)`).
        initial_feature_names (list of str): Names of the features, typically
            `adata.var_names`.
        n (int): Number of top features to return per principal component
            (default = 1).

    Returns:
        df (pandas.DataFrame): DataFrame with one row per principal component,
        listing the top contributing features.

    Example:
        Retrieve the top 5 features contributing to each PC:
            ```python
            from scpviz import utils as scutils
            pdata.pca(n_components=5)
            df = scutils.get_pca_importance(
                pdata.prot.uns['pca'],
                pdata.prot.var_names,
                n=5
            )
            ```
    """

    if isinstance(model, dict):
        pcs = np.asarray(model["PCs"])  # shape: n_components x n_features
    else:
        pcs = np.asarray(model.components_)  # shape: n_components x n_features

    n_pcs = pcs.shape[0]

    most_important = [
        np.abs(pcs[i]).argsort()[-n:][::-1] for i in range(n_pcs)
    ]
    most_important_names = [
        [initial_feature_names[idx] for idx in row] for row in most_important
    ]

    result = {
        f"PC{i + 1}": most_important_names[i] for i in range(n_pcs)
    }
    df = pd.DataFrame(result.items(), columns=["Principal Component", "Top Features"])
    return df

def get_protein_clusters(pdata, on='prot', layer='X', t=5, criterion='maxclust'):
    """
    Retrieve hierarchical clusters of proteins from stored linkage.

    This function uses linkage information stored in `pdata.stats` to
    partition proteins into clusters.

    Args:
        pdata (pAnnData): Input object containing `.stats` with clustering results.
        on (str): Data level to use, `"prot"` (default) or `"pep"`.
        layer (str): Data layer name used when the linkage was computed (default = `"X"`).
        t (int or float): Number of clusters (if `criterion="maxclust"`) or distance
            threshold for clustering.
        criterion (str): Clustering criterion passed to `scipy.cluster.hierarchy.fcluster`,
            e.g. `"maxclust"` or `"distance"`.

    Returns:
        clusters (dict): Mapping of `cluster_id → list of proteins`.
        None: If no linkage is found in `pdata.stats`.

    Note:
        Requires that a clustermap has been previously computed and linkage
        stored under `pdata.stats[f"{on}_{layer}_clustermap"]`.

    Related Functions:
        - plot_clustermap: Generates clustered heatmaps and stores linkage.
    """
    from scipy.cluster.hierarchy import fcluster
    
    key = f"{on}_{layer}_clustermap"
    stats = pdata.stats.get(key)
    if not stats or "row_linkage" not in stats:
        print(f"No linkage found for {key} in pdata.stats.")
        return None

    linkage = stats["row_linkage"]
    labels = fcluster(linkage, t=t, criterion=criterion)
    order = stats["row_order"]

    from collections import defaultdict
    clusters = defaultdict(list)
    for label, prot in zip(labels, order):
        clusters[label].append(prot)

    return dict(clusters)

# ----------------
# TO DOUBLE CHECK/THINK ABOUT...
def _map_uniprot_field(from_type: str, to_type):
    """
    Internal helper to resolve UniProt column names and required fields
    for a given identifier conversion request.

    Args:
        from_type (str): Source identifier type ('accession', 'gene').
        to_type (str or list of str): Target identifier type(s).
            Supported: 'gene', 'string', 'organism_id'.

    Returns:
        tuple: (from_col, to_cols, required_fields)
    """
    if isinstance(to_type, str):
        to_type = [to_type]

    # Validate allowed types
    valid_types = {"accession", "gene", "string", "organism_id"}
    if from_type not in valid_types:
        raise ValueError(f"Invalid from_type: '{from_type}'. Must be one of {valid_types}")
    if any(t not in valid_types for t in to_type):
        raise ValueError(f"Invalid to_type: {to_type}. Must be subset of {valid_types}")
    if from_type == "organism_id":
        raise ValueError("'organism_id' can only be used as a target (to_type).")

    field_map = {
        "accession": "accession",
        "gene": "gene_primary",
        "string": "xref_string",
        "organism_id": "organism_id",
    }

    from_col = field_map[from_type]
    to_cols = [field_map[t] for t in to_type]

    # Determine required UniProt fields for the query
    required_fields = set(["accession", from_col, *to_cols])
    if from_type == "gene":
        # Gene lookups usually need accession linkage
        required_fields |= {"accession"}

    return from_col, to_cols, list(required_fields)

def convert_identifiers(
    ids,
    from_type: str,
    to_type,
    pdata=None,
    use_cache: bool = True,
    return_type: str = "dict",
    verbose: bool = True,
):
    """
    Convert identifiers between UniProt-compatible types.

    Supports mapping between protein accessions, gene names, STRING IDs,
    and organism IDs. Multiple output types may be requested at once.

    Args:
        ids (list of str): Input identifiers.
        from_type (str): Source identifier type ('accession', 'gene').
            'organism_id' cannot be used as a source.
        to_type (str or list of str): Target identifier type(s).
            May include any of: ['gene', 'string', 'organism_id'].
        pdata (pAnnData, optional): pAnnData object providing cached
            accession–gene mappings. If provided, `use_cache` is
            automatically set to True.
        use_cache (bool): Whether to use cached mappings from `pdata`.
            (default: True)
        return_type (str): Output format:
            - 'dict': {input → {to_type → value}}
            - 'df': DataFrame with columns [from_type, *to_type]
            - 'both': (dict, DataFrame)
        verbose (bool): Whether to print progress messages.

    Returns:
        dict, pandas.DataFrame, or tuple: Depending on `return_type`.

    Example:
        >>> convert_identifiers(["P12345", "Q9XYZ1"], "accession", "gene", pdata=pdata)
        >>> convert_identifiers(["P12345"], "accession", ["gene", "string", "organism_id"], return_type="df")
    """
    import pandas as pd
    import numpy as np

    if not ids:
        empty_df = pd.DataFrame(columns=[from_type] + ([to_type] if isinstance(to_type, str) else list(to_type)))
        return {} if return_type != "df" else empty_df

    if pdata is not None:
        use_cache = True

    from_col, to_cols, search_fields = _map_uniprot_field(from_type, to_type)
    if isinstance(to_type, str):
        to_type = [to_type]

    # canonical UniProt field map (consistent with standardize_uniprot_columns)
    _FIELD_MAP = {
        "accession": "accession",
        "gene": "gene_primary",
        "string": "xref_string",
        "organism_id": "organism_id",
    }

    # --- Logging
    if verbose:
        print(f"{format_log_prefix('search', indent=1)} Converting from '{from_type}' to {to_type} for {len(ids)} identifiers...")
        if pdata is not None:
            cacheable_types = {"accession", "gene"}
            api_needed = [t for t in to_type if t not in cacheable_types]
            if set([from_type] + to_type).issubset(cacheable_types):
                print(f"{format_log_prefix('info_only', indent=2)} Using cached mapping from pdata (no UniProt queries).")
            elif api_needed:
                api_list = ", ".join(api_needed)
                print(f"{format_log_prefix('info_only', indent=2)} Using cached mapping for gene/accession; UniProt lookup required for: {api_list}.")
        else:
            print(f"{format_log_prefix('info_only', indent=2)} No pdata provided — querying UniProt for all target fields.")

    # --- Tier 1: cache lookup (only accession <-> gene)
    resolved = {id_: {t: None for t in to_type} for id_ in ids}
    to_query = list(ids)

    if pdata is not None and use_cache and {"accession", "gene"}.issuperset({from_type, *to_type}):
        if from_type == "accession" and "gene" in to_type:
            _, acc_to_gene = pdata.get_identifier_maps(on="protein")
            for acc in ids:
                if acc in acc_to_gene:
                    resolved[acc]["gene"] = acc_to_gene[acc]
        elif from_type == "gene" and "accession" in to_type:
            gene_to_acc, _ = pdata.get_identifier_maps(on="protein")
            for gene in ids:
                if gene in gene_to_acc:
                    resolved[gene]["accession"] = gene_to_acc[gene]

        # Filter unmapped
        to_query = [x for x, v in resolved.items() if not any(vv for vv in v.values())]

    # --- Tier 3: UniProt API
    df = pd.DataFrame()
    if len(to_query) > 0:
        # Hybrid case: gene → STRING / organism_id
        if from_type == "gene":
            gene_to_acc = convert_identifiers(to_query, "gene", "accession", pdata=pdata, use_cache=use_cache, verbose=False)
            accs = [v.get("accession") for v in gene_to_acc.values() if v.get("accession")]
            if accs:
                df = get_uniprot_fields(accs, search_fields=search_fields, standardize=True)
                df = standardize_uniprot_columns(df)
                df = df.drop_duplicates(subset="accession", keep="first")

                # Build per-target maps
                per_target_maps = {}
                for t in to_type:
                    col = _FIELD_MAP[t]
                    if col in df.columns:
                        per_target_maps[t] = dict(zip(df["accession"], df[col]))
                    else:
                        per_target_maps[t] = {}

                # Assign results
                for g, acc_dict in gene_to_acc.items():
                    acc = acc_dict.get("accession")
                    for t in to_type:
                        resolved[g][t] = per_target_maps[t].get(acc) if acc else None
            else:
                for g in to_query:
                    for t in to_type:
                        resolved[g][t] = None

        else:
            # Direct mapping (accession → X)
            df = get_uniprot_fields(to_query, search_fields=search_fields, standardize=True)

            # --- Clean up STRING results if present
            if not df.empty:
                if "xref_string" in df.columns and isinstance(df["xref_string"], pd.Series):
                    df["xref_string"] = (
                        df["xref_string"]
                        .astype(str)
                        .apply(lambda s: s.replace(";", "").strip() if isinstance(s, str) else np.nan)
                        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
                    )
                elif "string" in to_type and verbose:
                    print(f"{format_log_prefix('warn_only', indent=3)} UniProt did not return 'xref_string' field — possible API schema drift.")

            if not df.empty and from_col in df.columns:
                per_target_maps = {}
                for t in to_type:
                    col = _FIELD_MAP[t]
                    if col in df.columns:
                        per_target_maps[t] = dict(zip(df[from_col], df[col]))
                    else:
                        per_target_maps[t] = {}

                for id_ in to_query:
                    for t in to_type:
                        resolved[id_][t] = per_target_maps[t].get(id_)
            else:
                for id_ in to_query:
                    for t in to_type:
                        resolved[id_][t] = None

    # --- Reporting
    resolved_count = sum(
        any(vv is not None and not pd.isna(vv) for vv in v.values()) for v in resolved.values()
    )
    missing = [k for k, v in resolved.items() if all(vv is None or pd.isna(vv) for vv in v.values())]

    if verbose:
        local_resolved = len(ids) - len(to_query)
        api_resolved = resolved_count - local_resolved
        print(f"{format_log_prefix('result_only', indent=2)} {resolved_count}/{len(ids)} identifiers successfully converted "
            f"({local_resolved} local, {api_resolved} via UniProt).")
        if missing:
            print(f"{format_log_prefix('warn_only', indent=2)} {len(missing)} identifiers could not be resolved:")
            print("        " + ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""))

    # --- Output
    result_df = pd.DataFrame({from_type: list(resolved.keys())})
    for t in to_type:
        result_df[t] = [resolved[i][t] for i in result_df[from_type]]

    if return_type == "dict":
        return resolved
    elif return_type == "df":
        return result_df
    elif return_type == "both":
        return resolved, result_df
    else:
        raise ValueError("Invalid return_type. Choose from {'dict', 'df', 'both'}.")

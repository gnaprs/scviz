"""
This module contains functions for processing protein data and performing statistical tests.

Functions:
    get_samples: Get the sample names for the given class(es) type.
    protein_summary: Import protein data from an Excel file and summarize characteristics about each sample and sample groups.
    append_norm: Append normalized protein data to the original protein data.
    get_cv: Calculate the coefficient of variation (CV) for each case in the given data.
    get_abundance: Calculate the abundance of proteins across different groups.
    filter_group: Filter a DataFrame based on specified groups. Helper function for run_ttest.
    run_ttest: Run t-tests on specified groups in a DataFrame.
    get_abundance_query: Search and extract protein abundance data based on a specific list searching for accession, protein, description, pathway, or all.
    get_upset_contents: Get the contents for an UpSet plot based on the specified case list.
    ... more to come

Example:
    To use this module, import it and call functions from your code as follows:

    from scviz import utils
    data = pd.read_csv('data.csv')
    summarized_data = utils.protein_summary(data, variables=['region', 'amt'])
    

Todo:
    * Add corrections for differential expression.
    * Add more examples for each function.
    * Add functions to get GO enrichment and GSEA analysis (see GOATOOLS or STAGES).
"""

from typing import List, Optional, Dict, Any
from decimal import Decimal
from operator import index
from os import access
import re
import io
import requests
import warnings

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, chi2_contingency, fisher_exact
from scipy.sparse import csr_matrix
from sklearn.impute import SimpleImputer, KNNImputer

import upsetplot
import anndata as ad
from scviz import pAnnData

# Thoughts: functions that act on panndata and return only panndata should be panndata methods, utility functions should be in utils

# ----------------
# BASIC UTILITY FUNCTIONS

def log(msg, status="info"):
    prefix = {
        "info": "ℹ️",
        "warn": "⚠️",
        "ok": "✅",
        "fail": "❌",
        "check": "🔍"
    }.get(status, "")
    print(f"{prefix} {msg}")

# ----------------
# DATA PROCESSING FUNCTIONS
# NOTE: get_samplenames and get_classlist are very similar, may want to consider combining at some point
def get_samplenames(adata, classes):
    """
    Gets the sample names for the given class(es) type. Helper function for plot functions. 

    Parameters:
    - adata (anndata.AnnData): The AnnData object containing the sample names.
    - classes (str or list of str): The classes to use for selecting samples. E.g. 'cell_type' or ['cell_type', 'treatment'].

    Returns:
    - list of str: The sample names.

    Example:
    >>> samples = get_samplenames(adata, 'cell_type')
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
    Returns a list of unique classes for the given class(es) type. Helper function for plot functions.

    Parameters:
    - adata (anndata.AnnData): The AnnData object containing the classes.
    - classes (str or list of str): The classes to use for selecting samples. E.g. 'cell_type' or ['cell_type', 'treatment'].
    - order (list of str): The order to sort the classes in. Default is None.

    Returns:
    - list of str: The unique classes.

    Example:
    >>> classes = get_classlist(adata, classes = classes, order = order)
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

    Parameters:
    - adata: AnnData object
    - layer (str): Layer name or "X"

    Returns:
    - numpy.ndarray
    """
    if layer == "X":
        data = adata.X
    elif layer in adata.layers:
        data = adata.layers[layer]
    else:
        raise ValueError(f"Layer '{layer}' not found in .layers and is not 'X'.")

    return data.toarray() if hasattr(data, 'toarray') else data

def get_adata(pdata, on = 'protein'):
    if on == 'protein':
        return pdata.prot
    elif on == 'peptide':
        return pdata.pep
    else:
        raise ValueError("Invalid value for 'on'. Options are 'protein' or 'peptide'.")

def get_abundance(pdata, *args, **kwargs):
    """
    Wrapper to extract abundance from either pAnnData or AnnData.

    If pdata is a pAnnData object, calls pdata.get_abundance().
    If pdata is an AnnData object, calls the internal logic directly.

    Parameters:
        pdata: pAnnData or AnnData
        *args, **kwargs: passed to get_abundance

    Returns:
        pd.DataFrame
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
    Resolve gene or accession names to accession IDs from .var_names.

    Parameters:
        adata: AnnData or pAnnData object
        namelist: list of input names (genes or accessions)
        gene_col: column in .var containing gene names (default: "Genes")
        gene_map: optional precomputed map of gene → accession

    Returns:
        List of resolved accessions
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
        print("[resolve_accessions] Warning: Unmatched names:")
        for u in unmatched:
            print(f"  - {u}")

    return resolved

def get_upset_contents(pdata, classes, on = 'protein', upsetForm = True, debug=False):
    """
    Get the contents for an UpSet plot based on the specified case list. Helper function for UpSet plots.

    Parameters:
    - pdata (pAnnData): The pAnnData object containing the samples.
    - classes (str or list of str): The classes to use for selecting samples. E.g. 'cell_type' or ['cell_type', 'treatment'].
    - on (str): The data type to use for the UpSet plot. Options are 'protein' or 'peptide'. Default is 'protein'.

    Returns:
    - dict: The contents for the UpSet plot.
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

        # get proteins that are present in the filtered data (at least one value is not NaN)
        prot_present = data_filtered.var_names[(~np.isnan(data_filtered.X.toarray())).sum(axis=0) > 0]
        upset_dict[class_value] = prot_present.tolist()

    if upsetForm:
        upset_data = upsetplot.from_contents(upset_dict)
        return upset_data
    
    else:
        return upset_dict

def get_upset_query(upset_content, present, absent):
    prot_query = upsetplot.query(upset_content, present=present, absent=absent).data['id'].tolist()
    prot_query_df = get_uniprot_fields(prot_query)

    return prot_query_df

# TODO: check through all code and migrate to new filter_dict()
def filter(pdata, class_type, values, exact_cases = False, debug = False):
    """
    Filters samples from either a pAnnData or AnnData object using legacy-style
    (class_type + values) input.

    This function converts the input to the new dictionary-style format internally.
    - If `pdata` is a pAnnData object, this delegates to the class method `filter_sample_values()`.
    - If `pdata` is an AnnData object, filtering is performed directly on `.obs`.

    Note:
    For pAnnData users, it is recommended to use the class method `.filter_sample_values()`
    with dictionary-style input for cleaner and more consistent filtering.

    Parameters:
    - pdata (pAnnData or AnnData): The data object to filter.
    - class_type (str or list of str): Class labels to filter on (e.g. 'treatment' or ['treatment', 'cellline'])
    - values (list of str or list of list of str): Value(s) to match. For multiple class_types, this should be a list of values or a list of value-combinations. E.g. for cell_type: ['wt', 'kd'], or for a list of class_type [['wt', 'kd'],['control', 'treatment']].
    - exact_cases (bool): If True, treat values as exact combinations (AND across class types); otherwise allow OR logic within each class type.
    - debug (bool): If True, prints filter query.
    
    Returns:
    - Filtered object of the same type (pAnnData or AnnData).

    Example:
    >>> samples = filter(pdata, class_type="treatment", values="kd")

    >>> samples = utils.filter(adata, class_type = ['cell_type', 'treatment'], values = [['wt', 'kd'], ['control', 'treatment']])
    # returns samples where cell_type is either 'wt' or 'kd' and treatment is either 'control' or 'treatment'

    >>> samples = utils.filter(adata, exact_cases = True, class_type = ['cell_type', 'treatment'], values = [['wt', 'control'], ['kd', 'treatment']])
    # returns samples where cell_type is 'wt' and treatment is 'kd', or cell_type is 'control' and treatment is 'treatment'
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
    Converts legacy `classes` and `class_value` input into the new dictionary-style filter format.

    Parameters:
    - classes (str or list of str): Field name(s) used for filtering (e.g. 'treatment' or ['treatment', 'cellline']).
    - class_value (str, list of str, or list of list): The value(s) corresponding to the class(es).
        If a string, it may be underscore-joined (e.g. 'kd_AS').
        If a list of strings (e.g. ['kd_AS', 'sc_BE']), and exact_cases=True, each will be split and zipped with classes.
    - exact_cases (bool): Whether to return a list of exact match dictionaries or a combined OR filter.

    Returns:
    - dict or list of dicts: Formatted for dictionary-style filter input.
    """

    if isinstance(classes, str):
        # Simple case: one class
        if isinstance(class_value, list) and exact_cases:
            return [{classes: val} for val in class_value]
        else:
            return {classes: class_value}

    elif isinstance(classes, list):
        if exact_cases:
            # Handle the case where a single list is passed as one combination
            if isinstance(class_value, list) and all(isinstance(v, str) for v in class_value):
                if len(class_value) != len(classes):
                    raise ValueError("Length of class_value must match the number of classes.")
                return [{cls: val for cls, val in zip(classes, class_value)}]

            if isinstance(class_value, str):
                class_value = [class_value]

            formatted = []
            for entry in class_value:
                if isinstance(entry, str):
                    values = entry.split('_')
                else:
                    values = entry
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
    Helper to resolve (classes, class_value) inputs and apply filtering.

    Parameters:
    - adata (AnnData or pAnnData): The data object to filter.
    - classes (str, list, or None): Class label(s) for filtering.
    - class_value (str or list): Corresponding value(s) to filter.
    - debug (bool): Print resolved class/value pairs.
    - filter_func (callable): Filtering function to use (default: `utils.filter`).

    Returns:
    - Filtered data object (same type as input).
    """

    from scviz import utils  # safe to do here in case of circular import

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
    Returns the column name in .pep.var that maps peptides to proteins, or Series of mapping 
    based on the data source recorded in pdata.metadata.

    Parameters:
        pdata : pAnnData
            The annotated proteomics object with .metadata and .pep
        return_series : bool, optional (default=False)
            If True, returns a Series mapping peptides to proteins.
            If False, returns the name of the mapping column

    Returns:
        str : column name containing peptide-to-protein mapping
        or pd.Series : mapping of peptides to proteins

    Raises:
        ValueError if source is unrecognized or mapping column not found.
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
# EXPLORATION FUNCTIONS
def get_uniprot_fields_worker(prot_list, search_fields=['accession', 'id', 'protein_name', 'gene_primary', 'gene_names', 'go', 'go_f' ,'go_f', 'go_c', 'go_p', 'cc_interaction'], verbose = False):
    """ Worker function to get data from Uniprot for a list of proteins, used by get_uniprot_fields(). Calls to UniProt REST API, and returns batch of maximum 1024 proteins at a time.

    Args:
        prot_list (list): list of protein IDs
        search_fields (list): list of fields to search for.

        For more information, see https://www.uniprot.org/help/return_fields for list of search_fields.
        Current function accepts accession protein list. For more queries, see https://www.uniprot.org/help/query-fields for a list of query fields that can be searched for.

    Returns:
        pandas.DataFrame: DataFrame with the results
    """

    base_url = 'https://rest.uniprot.org/uniprotkb/stream'
    fields = "%2C".join(search_fields)
    query_parts = ["%28accession%3A" + id + "%29" for id in prot_list]
    query = "+OR+".join(query_parts)
    query = "%28" + query + "%29"
    format_type = 'tsv'
    
    if verbose:
        print(f"Querying Uniprot for {len(prot_list)} proteins")

    # full url
    url = f'{base_url}?fields={fields}&format={format_type}&query={query}'
    
    results = requests.get(url)
    results.raise_for_status()
    
    df = pd.read_csv(io.StringIO(results.text), sep='\t')

    isoform_prots = set(prot_list) - set(df['Entry'])
    if isoform_prots:
        # print statement for missing proteins, saying will now search for isoforms
        if verbose:
            print(f'Searching for isoforms in unmapped accessions: {len(isoform_prots)}')
        # these are typically isoforms, so we will try to search for them again
        query_parts = ["%28accession%3A" + id + "%29" for id in isoform_prots]
        query = "+OR+".join(query_parts)
        query = "%28" + query + "%29"
        url = f'{base_url}?fields={fields}&format={format_type}&query={query}'
        
        results = requests.get(url)
        results.raise_for_status()
        
        isoform_df = pd.read_csv(io.StringIO(results.text), sep='\t')
        
        # now, let's search isoforms by entry name (id) instead
        isoform_entrynames = isoform_df['Entry Name']
        query_parts = ["%28id%3A" + id + "%29" for id in isoform_entrynames]
        query = "+OR+".join(query_parts)
        query = "%28" + query + "%29"
        url = f'{base_url}?fields={fields}&format={format_type}&query={query}'

        results = requests.get(url)
        results.raise_for_status()

        isoform_df2 = pd.read_csv(io.StringIO(results.text), sep='\t')
        entry_mapping = {entry.split('-')[0]: (entry, protein_name) for entry, protein_name in zip(isoform_df['Entry'], isoform_df['Protein names'])}
        isoform_df2[['Entry', 'Protein names']] = isoform_df2['Entry'].apply(lambda x: entry_mapping.get(x.split('-')[0], (x, None))).apply(pd.Series)
        
        df = pd.concat([df, isoform_df2], axis=0)
        
        # what's left is missing proteins
        missing_prots = set(prot_list) - set(df['Entry'])
        if missing_prots:
            print(f"Proteins not found in uniprot database after two attempts: {missing_prots}")
            missing_df = pd.DataFrame(missing_prots, columns=['Entry'])
            for col in df.columns:
                if col != 'Entry':
                    missing_df[col] = np.nan
            df = pd.concat([df, missing_df], axis=0)
    
    return df

def get_uniprot_fields(prot_list, search_fields=['accession', 'id', 'protein_name', 'gene_primary', 'gene_names', 'go', 'go_f' ,'go_f', 'go_c', 'go_p', 'cc_interaction'], batch_size=1024, verbose=False):
    """ Get data from Uniprot for a list of proteins. Uses get_uniprot_fields_worker to get data in batches of batch_size. For more information, see https://www.uniprot.org/help/return_fields for list of search_fields.
        Current function accepts accession protein list. For more queries, see https://www.uniprot.org/help/query-fields for a list of query fields that can be searched for.

    Args:
        prot_list (list): list of protein IDs
        search_fields (list): list of fields to search for.
        batch_size (int): number of proteins to search for in each batch. Default (and maximum) is 1024.

    Returns:
        pandas.DataFrame: DataFrame with the results
    
    Example:
        >>> uniprot_list = ["P40925", "P40926"]
        >>> df = get_uniprot_fields(uniprot_list)
    """

    # BUGFIX: accession should be first in the list of search fields, otherwise error thrown in worker function
    if 'accession' not in search_fields:
        search_fields = ['accession'] + search_fields
    elif search_fields[0] != 'accession':
        search_fields.remove('accession')
        search_fields = ['accession'] + search_fields
    # Split the id_list into batches of size batch_size
    batches = [prot_list[i:i + batch_size] for i in range(0, len(prot_list), batch_size)]
    # print total number and number of batches
    if verbose:
        print(f"Total number of proteins: {len(prot_list)}, Number of batches: {len(batches)}")
        print(f"Fields: {search_fields}")
    # Initialize an empty dataframe to store the results
    full_method_df = pd.DataFrame()
    
    # Loop through each batch and get the uniprot fields
    for batch in batches:
        # print progress
        if verbose:
            print(f"Processing batch {batches.index(batch) + 1} of {len(batches)}")
        batch_df = get_uniprot_fields_worker(batch, search_fields)
        full_method_df = pd.concat([full_method_df, batch_df], ignore_index=True)
    
    return full_method_df

# ----------------
# STATISTICAL TEST FUNCTIONS
def pairwise_log2fc(data1, data2):
    """
    Compute pairwise median log2FC for each feature between two sample groups. Helper function for pdata.de() in fold-change options pairwise_median and pep_pairwise_median.

    Parameters:
    - data1, data2: np.ndarray of shape (n_samples, n_features)

    Returns:
    - median_log2fc: np.ndarray of shape (n_features,)
    """
    n1, n2 = data1.shape[0], data2.shape[0]

    # data1[:, None, :] has shape (n1, 1, n_features)
    # data2[None, :, :] has shape (1, n2, n_features)
    # The result is an array of shape (n1, n2, n_features)
    with np.errstate(divide='ignore', invalid='ignore'):
        pairwise_ratios = np.log2(data1[:, None, :] / data2[None, :, :])  # (n1, n2, features)

    # Reshape to (n1*n2, n_features) and compute the median along the first axis.
    median_fc = np.nanmedian(pairwise_ratios.reshape(-1, data1.shape[1]), axis=0)
    return median_fc

# TODO: fix with pdata.summary maybe call stats_ttest instead
def run_summary_ttest(protein_summary_df, test_variables, test_pairs, print_results=False, test_variable='total_count'):
    """
    Run t-tests on specified groups in a DataFrame returned from get_protein_summary.

    Args:
        protein_summary_df (pd.DataFrame): The DataFrame returned from get_protein_summary.
        test_variables (list): The variables to use for grouping.
        test_pairs (list): The pairs of groups to compare.
        print_results (bool, optional): Whether to print the t-test results. Defaults to False.
        test_variable (str, optional): The variable to test for in the t-test. Acceptable variables are any of the columns in df_files. Defaults to 'total_count'.

    Returns:
        list: A list of t-test results for each pair of groups. Each result is a tuple containing the t-statistic and the p-value.

    Example usage:
        >>> test_variables = ['region','amt']
        >>> test_pairs = [[['cortex','sc'], ['cortex','20000']], 
                        [['cortex','20000'], ['snpc','10000']], 
                        [['mp_cellbody','6000'], ['mp_axon','6000']]]
        >>> ttestparams = run_summary_ttest(protein_summary_df, test_variables, test_pairs, test_variable='total_count')
    """
    # check if every pair in test_pairs is a list of length 2, else throw error message
    if not all(len(pair) == 2 for pair in test_pairs):
        raise ValueError("Error: Each pair in test_pairs must contain two groups (e.g. compare [['circle','big'] and ['square','small']])")

    # check if every element in test_pairs[i][0] and test_pairs[i][1] is the same length as test_variables, else throw error message
    if not all(len(pair[0]) == len(test_variables) and len(pair[1]) == len(test_variables) for pair in test_pairs):
        raise ValueError("Error: Each group in each pair in test_pairs must match the length of test_variables")

    ttest_params = []
    for pair in test_pairs:
        group1 = filter_by_group(protein_summary_df, test_variables, pair[0])
        group2 = filter_by_group(protein_summary_df, test_variables, pair[1])

        t_stat, p_val = ttest_ind(group1[test_variable], group2[test_variable])
        if print_results:
            print(f"Testing for {test_variable} between pair {pair[0]} and {pair[1]}:")
            print(f"N1: {len(group1)}, N2: {len(group2)}")
            print(f"t-statistic: {t_stat}, p-value: {p_val}") 
        ttest_params.append([pair[0], pair[1], t_stat, p_val, len(group1), len(group2)])

    ttest_df = pd.DataFrame(ttest_params, columns=['Group1', 'Group2', 'T-statistic', 'P-value', 'N1', 'N2'])
    return ttest_df

def get_pca_importance(model, initial_feature_names, n=1):
    """
    Get the most important feature for each principal component in a PCA model.

    Args:
        model (sklearn.decomposition.PCA): The PCA model.
        initial_feature_names (list): The initial feature names. Typically adata.var_names.
        n (int): The number of top features to return for each principal component.

        
    Returns:
        df (pd.DataFrame): A DataFrame containing the most important feature for each principal component.

    Example:
        >>> from scviz import utils as scutils
        >>> pca = PCA(n_components=5)
        >>> pca.fit(data)
        >>> df = scutils.get_pca_importance(pca)
    """

    # number of components
    n_pcs= model['PCs'].shape[0]

    # get the index of the most important feature on EACH component
    most_important = [np.abs(model['PCs'][i]).argsort()[-n:][::-1] for i in range(n_pcs)]
    most_important_names = [[initial_feature_names[idx] for idx in most_important[i]] for i in range(n_pcs)]


    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
    df = pd.DataFrame(dic.items(), columns=['Principal Component', 'Top Features'])

    return df

# ----------------
# TO FIX
# TODO: fix
def get_query(pdata, search_term, search_on = 'gene', on = 'protein'):
    valid_search_terms = ['gene', 'protein', 'description', 'pathway', 'all']

    # If search is a list, remove duplicates and check if it includes all terms
    if isinstance(search, list):
        search = list(set(search))  # Remove duplicates
        if set(valid_search_terms[:-1]).issubset(set(search)):  # Check if it includes all terms
            print('All search terms included. Using search term \'all\'.')
            search = 'all'
        elif not all(term in valid_search_terms for term in search):  # Check if all terms are valid
            raise ValueError(f'Invalid search term. Please use one of the following: {valid_search_terms}')
    # If search is a single term, check if it's valid
    elif search not in valid_search_terms:
        raise ValueError(f'Invalid search term. Please use one of the following: {valid_search_terms}')

    if on == 'protein':
        adata = pdata.prot
    elif on == 'peptide':
        adata = pdata.pep

    if search_on == 'gene':
        return adata[adata.var['Gene Symbol'] == search_term]
    elif search_on == 'protein':
        return adata[adata.var['Accession'] == search_term]
    elif search_on == 'description':
        return adata[adata.var['Description'] == search_term]
    elif search_on == 'pathway':
        return adata[adata.var['WikiPathways'] == search_term]

# TODO: fix to work with panndata, just need to search through .vars and identify whether keyword is in any column
def get_abundance_query(pdata, cases, genelist, search='gene', on = 'protein'):
    """
    Search and extract protein abundance data based on a specific gene list.

    This function searches and extracts protein abundance data for specified cases based on a specific gene list. The search can be performed on 'gene', 'protein', 'description', 'pathway', or 'all'. It also accepts a list of terms.

    Args:
        pdata (pandas.DataFrame): The protein data.
        cases (list): The cases to include in the search.
        genelist (list): The genes to include in the search. Can also be accession numbers, descriptions, or pathways.
        search (str): The search term to use. Can be 'gene', 'protein', 'description', 'pathway', or 'all'. Also accepts list of terms.

    Returns:
        matched_features_data (pandas.DataFrame): Extracted protein abundance data, along with matched search features and the respective genes they were matched to.
        combined_abundance_data (pandas.DataFrame): Protein abundance data per sample for matching genes.

    Raises:
        ValueError: If the search term is not valid. Valid search terms are 'gene', 'protein', 'description', 'pathway', 'all', or a list of these.

    Example:
        >>> from scviz import utils as scutils
        >>> import pandas as pd
        >>> cases = [['head'],['heart'],['tail']]
        >>> matched_features, combined_abundance = scutils.get_abundance_query(data, cases, gene_list.Gene, search=["gene","pathway","description"])
    """

    # TODO: WILL NEED TO USE get_uniprot_fields TO GET ANNOTATED DATA

    valid_search_terms = ['gene', 'protein', 'description', 'pathway', 'all']

    # If search is a list, remove duplicates and check if it includes all terms
    if isinstance(search, list):
        search = list(set(search))  # Remove duplicates
        if set(valid_search_terms[:-1]).issubset(set(search)):  # Check if it includes all terms
            print('All search terms included. Using search term \'all\'.')
            search = 'all'
        elif not all(term in valid_search_terms for term in search):  # Check if all terms are valid
            raise ValueError(f'Invalid search term. Please use one of the following: {valid_search_terms}')
    # If search is a single term, check if it's valid
    elif search not in valid_search_terms:
        raise ValueError(f'Invalid search term. Please use one of the following: {valid_search_terms}')

    # ------------------------------------------------------------------------------------------------
    data_abundance = data.copy()
    data = data.copy()

    for case in cases:
        vars = ['Abundance: '] + case
        append_string = '_'.join(vars[1:])
        cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]
        data['Average: '+append_string] = data[cols].mean(axis=1, skipna=True)
        data['Stdev: '+append_string] = data[cols].std(axis=1, skipna=True)

    case_col = [data.columns.get_loc('Average: '+'_'.join(case)) for case in cases]

    search_to_column = {
        'gene': 'Gene Symbol',
        'protein': 'Accession',
        'description': 'Description',
        'pathway': 'WikiPathways',
        'all': ['Gene Symbol', 'Accession', 'Description', 'WikiPathways']
    }

    # search can be a single term or a list of terms
    columns = [search_to_column[term] for term in (search if isinstance(search, list) else [search]) if term != 'all']
    if 'all' in (search if isinstance(search, list) else [search]):
        columns.extend(search_to_column['all'])

    for column in columns:
        data[f'Matched in {column}'] = data[column].apply(lambda x: [])

        for gene in genelist:
            regex = rf"\b{re.escape(gene)}\b"
            data[f'Matched in {column}'] = data.apply(lambda row: row[f'Matched in {column}'] + [gene] if isinstance(row[column], str) and re.search(regex, row[column], re.IGNORECASE) else row[f'Matched in {column}'], axis=1)

    data = data[data.filter(regex='Matched').apply(any, axis=1)]
    data.set_index('Gene Symbol', inplace=True)

    if data.shape[0] == 0:
        raise ValueError('No data to plot. Please check the gene list and the search parameters.')

    num_new_cols = len(search) if isinstance(search, list) else (4 if search == 'all' else 1)
    case_col.extend(range(len(data.columns) - (num_new_cols-1), len(data.columns) + 1))
    matched_features_data = data.iloc[:,[i-1 for i in case_col]]
    matched_features_data = matched_features_data.dropna(how='all')

    data_abundance.set_index('Gene Symbol', inplace=True)
    data_abundance = data_abundance.loc[data.index]
    combined_abundance_data = pd.DataFrame()

    for case in cases:
        vars = ['Abundance: '] + case
        cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]
        combined_abundance_data = pd.concat([combined_abundance_data, data_abundance[cols]], axis=1)

    return matched_features_data, combined_abundance_data

# TODO: sync with get_uniprot_fields
def convert_identifiers(input_list, input_type, output_type, df):
    """
    Convert a list of gene information of a certain type to another type.

    Args:
        input_list (list): The list of gene information to convert.
        input_type (str): The type of the input gene information. Must be one of 'gene_symbol', 'accession', or 'gene_id'.
        output_type (str): The type of the output gene information. Must be one of 'gene_symbol', 'accession', or 'gene_id'.
        df (pd.DataFrame): The DataFrame containing the gene information.

    Returns:
        output_list (list): The converted list of gene information.
    """
    if input_type not in ['gene_symbol', 'accession', 'gene_id'] or output_type not in ['gene_symbol', 'accession', 'gene_id']:
        raise ValueError("input_type and output_type must be one of 'gene_symbol', 'accession', or 'gene_id'")

    convert_dict = {
        'gene_symbol': 'Gene Symbol',
        'accession': 'Accession',
        'gene_id': 'Gene ID'
    }

    input_type = convert_dict[input_type]
    output_type = convert_dict[output_type]
        
    output_list = df.loc[df[input_type].isin(input_list), output_type].tolist()

    return output_list

# TODO: add function to get GO enrichment, GSEA analysis (see GOATOOLS or STAGES or Enrichr?)
# TO INTEGRATE
def get_string_id(gene,species = 9606):
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "tsv-no-header"
    method = "get_string_ids"
    
    params = {
        "identifiers" : "\r".join(gene),
        "species" : species, 
        "limit" : 1, 
        "echo_query" : 1,
    }
    
    request_url = "/".join([string_api_url, output_format, method])
    results = requests.post(request_url, data=params)
    s_id = []
    for line in results.text.strip().split("\n"):
        l = line.split("\t")
        try:
            string_id = l[2]
            s_id.append(string_id)
        except:
            continue
    
    return s_id

# TO INTEGRATE
def get_string_annotation(gene,universe,species = 9606):
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "json"
    method = "enrichment"

    params = {
        "identifiers" : "%0d".join(gene),
        # "background_string_identifiers": "%0d".join(universe),
        "species" : species
    }
    
    request_url = "/".join([string_api_url, output_format, method])
    results = requests.post(request_url, data=params)
    print(results.text)
    try:
        annotation = pd.read_json(results.text)
    except:
        annotation = pd.DataFrame()
    return annotation

# TO INTEGRATE
def get_string_network(gene,comparison,species = 9606):
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "highres_image"
    method = "network"
    
    if len(gene) <= 10:
        hide_label = 0
    else:
        hide_label = 1
    
    params = {
        "identifiers" : "%0d".join(get_string_id(gene)),
        "species" : species,
        "required_score" : 700,
        "hide_disconnected_nodes" : hide_label,
        "block_structure_pics_in_bubbles" : 1,
        "flat_node_design" : 1,
        "center_node_labels" : 1
    }
    
    request_url = "/".join([string_api_url, output_format, method])
    response = requests.post(request_url, data=params)
    
    with open(f'{comparison}.png','wb') as file:
        file.write(response.content)
    
    return True
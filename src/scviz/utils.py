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
# DATA PROCESSING FUNCTIONS
def get_samplenames(adata, class_type):
    """
    Gets the sample names for the given class(es) type. Helper function for plot functions. 

    Parameters:
    - adata (anndata.AnnData): The AnnData object containing the sample names.
    - class_type (str or list of str): The classes to use for selecting samples. E.g. 'cell_type' or ['cell_type', 'treatment'].

    Returns:
    - list of str: The sample names.

    Example:
    >>> samples = get_samplenames(adata, 'cell_type')
    """

    if class_type is None:
        return None
    elif isinstance(class_type, str):
        return adata.obs[class_type].values.tolist()
    elif isinstance(class_type, list):
        return adata.obs[class_type].apply(lambda row: ', '.join(row.values.astype(str)), axis=1).values.tolist()
    else:
        raise ValueError("Invalid input for 'class_type'. It should be None, a string, or a list of strings.")
    
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

def get_adata(pdata, on = 'protein'):
    if on == 'protein':
        return pdata.prot
    elif on == 'peptide':
        return pdata.pep
    else:
        raise ValueError("Invalid value for 'on'. Options are 'protein' or 'peptide'.")

def get_upset_contents(pdata, classes, on = 'protein', upsetForm = True):
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
        if classes is None:
            values = class_value.split('_')
            # print(f'Classes: {classes}, Values: {values}') if debug else None
            data_filter = filter(adata, classes, values, suppress_warnings=True)
        elif isinstance(classes, str):
            # print(f'Class: {classes}, Value: {class_value}') if debug else None
            data_filter = filter(adata, classes, class_value, suppress_warnings=True)
        elif isinstance(classes, list):
            values = class_value.split('_')
            # print(f'Classes: {classes}, Values: {values}') if debug else None
            data_filter = filter(adata, classes, values, suppress_warnings=True)

        # get proteins that are present in the filtered data (at least one value is not NaN)
        prot_present = data_filter.var_names[(~np.isnan(data_filter.X.toarray())).sum(axis=0) > 0]
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

# IMPORTANT: move to class function, ensure nothing else breaks
def filter(pdata, class_type, values, exact_cases = False, suppress_warnings = False):
    """
    Filters out for the given class(es) type. Returns a copy of the filtered pdata object, does not modify the original object.

    Parameters:
    - pdata (pAnnData): The pAnnData object containing the samples.
    - class_type (str or list of str): The classes to use for selecting samples. E.g. 'cell_type' or ['cell_type', 'treatment'].
    - values (list of str or list of list of str): The values to select for within the class_type. E.g. for cell_type: ['wt', 'kd'], or for a list of class_type [['wt', 'kd'],['control', 'treatment']].
    - exact_cases (bool): Whether to match the exact cases specified by the user. Default is False.
    
    Returns:
    - pAnnData: Returns a copy of the filtered pdata object. Does not modify the original object.

    Example:
    >>> samples = get_samples(adata, class_type = ['cell_type', 'treatment'], values = [['wt', 'kd'], ['control', 'treatment']])
    # returns samples where cell_type is either 'wt' or 'kd' and treatment is either 'control' or 'treatment'

    >>> samples = get_samples(adata, exact_cases = True, class_type = ['cell_type', 'treatment'], values = [['wt', 'control'], ['kd', 'treatment']])
    # returns samples where cell_type is 'wt' and treatment is 'kd', or cell_type is 'control' and treatment is 'treatment'
    """
    pdata = pdata.copy()

    is_anndata = False
    if isinstance(pdata, ad.AnnData):
        if not suppress_warnings:
            print("Warning: The provided object is an AnnData object, not a pAnnData object. Proceeding with the filter.")
        is_anndata = True
    elif not isinstance(pdata, pAnnData.pAnnData):
        raise ValueError("Invalid input for 'pdata'. It should be a pAnnData object.")

    if class_type is None:
        print("No class type specified. Returning unmodified pAnnData object.")
        return pdata
    
    if isinstance(class_type, str):
        query = f"(adata.obs['{class_type}'] == '{values}')"
        print('DEVELOPMENT: testing for query - ', query)
    elif isinstance(class_type, list):
        # if values is not wrapped in a list, wrap it
        if len(values) != 1:
            values = [values]
            
        if exact_cases:
            query = " | ".join([
                " & ".join(["(adata.obs['{}'] == '{}')".format(cls, val) for cls, val in zip(class_type, (vals if isinstance(vals, list) else [vals]))]) for vals in values
            ])
            print('DEVELOPMENT: exact|testing for query - ', query)
        else:
            # query = " & ".join([
            #     "({})".format(' | '.join(["(adata.obs['{}'] == '{}')".format(cls, val) for val in vals])) for cls, vals in zip(class_type, values)])
            query = " & ".join([
                "({})".format(' | '.join(["(adata.obs['{}'] == '{}')".format(cls, val) for val in (vals if isinstance(vals, list) else [vals])])) for cls, vals in zip(class_type, values)
            ])
            print('DEVELOPMENT: non-exact|testing for query - ', query)
            
    else:
        raise ValueError("Invalid input for 'class_type'. It should be None, a string, or a list of strings.")
    
    if is_anndata:
        adata = pdata
        pdata = adata[eval(query)]
    else:
        if pdata.prot is not None:
            adata = pdata.prot
            pdata.prot = adata[eval(query)]
        if pdata.pep is not None:
            adata = pdata.pep
            pdata.pep = adata[eval(query)]
        pdata._update_summary()
        pdata._append_history(f"Filtered by class type: {class_type}, values: {values}, exact_cases: {exact_cases}. Copy of the filtered pAnnData object returned.")    

    return pdata

# ----------------
# EXPLORATION(?) FUNCTIONS
def get_uniprot_fields_worker(prot_list, search_fields=['accession', 'id', 'protein_name', 'gene_primary', 'gene_names', 'go', 'go_f' ,'go_f', 'go_c', 'go_p', 'cc_interaction']):
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
    
    print(f"Querying Uniprot for {len(prot_list)} proteins")

    # full url
    url = f'{base_url}?fields={fields}&format={format_type}&query={query}'
    
    results = requests.get(url)
    results.raise_for_status()
    
    df = pd.read_csv(io.StringIO(results.text), sep='\t')

    isoform_prots = set(prot_list) - set(df['Entry'])
    if isoform_prots:
        # print statement for missing proteins, saying will now search for isoforms
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

def get_uniprot_fields(prot_list, search_fields=['accession', 'id', 'protein_name', 'gene_primary', 'gene_names', 'go', 'go_f' ,'go_f', 'go_c', 'go_p', 'cc_interaction'], batch_size=1024):
    """ Get data from Uniprot for a list of proteins. Uses get_uniprot_fields_worker to get data in batches of batch_size.

    Args:
        prot_list (list): list of protein IDs
        search_fields (list): list of fields to search for.
        batch_size (int): number of proteins to search for in each batch. Default (and maximum) is 1024.

        For more information, see https://www.uniprot.org/help/return_fields for list of search_fields.
        Current function accepts accession protein list. For more queries, see https://www.uniprot.org/help/query-fields for a list of query fields that can be searched for.

    Returns:
        pandas.DataFrame: DataFrame with the results
    
    Example:
        >>> uniprot_list = ["P40925", "P40926"]
        >>> df = get_uniprot_fields(uniprot_list)
    """

    # Split the id_list into batches of size batch_size
    batches = [prot_list[i:i + batch_size] for i in range(0, len(prot_list), batch_size)]
    # print total number and number of batches
    print(f"Total number of proteins: {len(prot_list)}, Number of batches: {len(batches)}")
    print(f"Fields: {search_fields}")
    # Initialize an empty dataframe to store the results
    full_method_df = pd.DataFrame()
    
    # Loop through each batch and get the uniprot fields
    for batch in batches:
        # print progress
        print(f"Processing batch {batches.index(batch) + 1} of {len(batches)}")
        batch_df = get_uniprot_fields_worker(batch, search_fields)
        full_method_df = pd.concat([full_method_df, batch_df], ignore_index=True)
    
    return full_method_df

# ----------------
# STATISTICAL TEST FUNCTIONS
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
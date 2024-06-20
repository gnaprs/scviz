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
from operator import index
from os import access
import pandas as pd
import numpy as np
import re
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, chi2_contingency, fisher_exact
from decimal import Decimal
from upsetplot import from_contents

import anndata as ad
from scipy.sparse import csr_matrix
from sklearn.impute import SimpleImputer, KNNImputer
from scviz import pAnnData



# Thoughts: functions that act on panndata and return only panndata should be panndata methods, utility functions should be in utils

def get_samplenames(adata,class_type):
    """
    Get the sample names for the given class(es) type.

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
    
def filter(pdata, class_type, values, exact_cases = False):
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
        print("Warning: The provided object is an AnnData object, not a pAnnData object. Proceeding with the filter.")
        is_anndata = True
    elif not isinstance(pdata, pAnnData.pAnnData):
        raise ValueError("Invalid input for 'pdata'. It should be a pAnnData object.")

    if class_type is None:
        print("No class type specified. Returning unmodified pAnnData object.")
        return pdata
    
    if isinstance(class_type, str):
        if exact_cases:
            query = " | ".join([f"(adata.obs['{class_type}'] == '{val}')" for val in values])
            print(query)
        else:
            query = " | ".join([f"(adata.obs['{class_type}'] == '{val}')" for val in values])
            print(query)
    elif isinstance(class_type, list):
        if exact_cases:
            query = " | ".join([" & ".join(["(adata.obs['{}'] == '{}')".format(cls, val) for cls, val in zip(class_type, vals)]) for vals in values])
            print(query)
        else:
            query = " & ".join(["({})".format(' | '.join(["(adata.obs['{}'] == '{}')".format(cls, val) for val in vals])) for cls, vals in zip(class_type, values)])
            print(query)
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
        pdata._summary()

    return pdata

# !TODO: move to class function | for peptide, use 'Annotated Sequence' instead of 'Accession' and shared search subset='Annotated Sequence'
def get_cv(data, cases, variables=['region', 'amt'], sharedPeptides = False):
    """
    Calculate the coefficient of variation (CV) for each case in the given data.

    This function calculates the CV for each case in the given data. The cases and variables to consider when calculating CV can be specified. There is also an option to calculate CV for only shared peptides identified across all cases.

    Args:
        data (pandas.DataFrame): The input data containing the CV values.
        cases (list of lists): The cases to calculate CV for. Each case is a list of values corresponding to the variables.
        variables (list, optional): The variables to consider when calculating CV. Default is ['region', 'amt'].
        sharedPeptides (bool, optional): Whether to calculate CV for only shared peptides identified across all cases. Default is False.

    Returns:
        cv_df (pandas.DataFrame): The DataFrame containing the CV values for each case, along with the corresponding variable values.

    Raises:
        None

    Example:
        >>> import scviz
        >>> sample_types = [[i,j] for i in ['a','b','c'] for j in [1,2,3]]
        >>> cv_df = scviz.utils.get_cv(data, sample_types, variables=['letters','numbers'])
    """
    # check if the len of each element in cases have the same length as len(variables), else throw error message
    if not all(len(cases[i]) == len(variables) for i in range(len(cases))):
        print("Error: length of each element in cases must be equal to length of variables")
        return
    
    # make dataframe for each case with all CV values
    cv_df = pd.DataFrame()
    data = data.copy()

    #! TODO: consider calculating CVs from scratch instead of using the CV values in the data
    for j in range(len(cases)):
        vars = ['CV'] + cases[j]
        cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]
        nsample = len(cols)

        # merge all CV columns into one column
        X = np.zeros((nsample*len(data)))
        accessions = []
        for i in range(nsample):
            X[i*len(data):(i+1)*len(data)] = data[cols[i]].values
            accessions=accessions+data['Accession'].values.tolist()
        # remove nans
        accessions = [accessions[i] for i in range(len(accessions)) if not np.isnan(X[i])]
        X = X[~np.isnan(X)]/100

        # add X to cur_df, and append case info of enzyme, method and amt to each row
        cur_df = pd.DataFrame()
        cur_df['cv'] = X
        cur_df['accession'] = accessions
        for i in range(len(variables)):
            cur_df[variables[i]] = cases[j][i]

        # append cur_df to cv_df
        cv_df = pd.concat([cv_df, cur_df], ignore_index=True)

        if sharedPeptides:
            cv_df['shared'] = cv_df.duplicated(subset='accession', keep=False)

    return cv_df

# TODO: DEPRECATED, now as get_samples()
def get_abundance(data: pd.DataFrame, cases: List[List[str]], prot_list: Optional[List[str]] = None, list_type: str = 'accession',abun_type: str = 'average') -> Dict[str, Any]:
    """
    Returns the abundance of proteins in the given data.

    This function calculates the abundance of proteins for specified cases in the given data. The abundance can be calculated as 'average' or 'raw'. Optionally, the data can be filtered by protein names.

    Args:
        data (pandas.DataFrame): The protein data.
        cases (list of lists): The cases to return abundances for.
        prot_list (list, optional): List of accession names to filter the data to just specified proteins. Default is None.
        abun_type (str, optional): Type of abundance calculation to perform ('average' or 'raw'). Default is 'average'. When raw, function returns list of [abundance, accession].

    Returns:
        abun_dict (dict): Dictionary containing the abundance values and ranks for each case.

    Raises:
        None
    """
    if list_type == 'accession':
        index = 'Accession'
    elif list_type == 'gene':
        index = 'Gene Symbol'
    else:
        raise ValueError('Invalid list type. Please use either "accession" or "gene".')

    if abun_type=='average':
        # create empty list to store abundance values
        abun_dict = {}
        data = data.copy()
        # extract columns that contain the abundance data for the specified case
        for j in range(len(cases)):
            vars = ['Abundance: '] + cases[j]

            if prot_list is not None:
                # extract out rows where list is in prot_list
                data = data[data[index].isin(prot_list)]

            cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]

            if not cols:
                raise ValueError("No columns found. Please verify that the input 'cases' matches the columns in the data.")

            # concat elements 1 till end of vars into one string
            append_string = '_'.join(vars[1:])

            # average abundance of proteins across these columns, ignoring NaN values
            data['Average: '+append_string] = data[cols].mean(axis=1, skipna=True)
            data['Stdev: '+append_string] = data[cols].std(axis=1, skipna=True)

            # sort by average abundance
            data.sort_values(by=['Average: '+append_string], ascending=False, inplace=True)
            abundance = (data['Average: '+append_string]
                        .rename('Average')
                        .to_frame()
                        .assign(Rank=np.arange(1, len(data)+1)))
            
            abundance.set_index(data['Accession'], inplace=True)
            abun_dict[append_string] = abundance
        return abun_dict

    elif abun_type == 'raw':
        # create empty list to store abundance values
        abun_dict = {}
        data = data.copy()
        # extract columns that contain the abundance data for the specified method and amount
        for j in range(len(cases)):
            vars = ['Abundance: '] + cases[j]

            if prot_list is not None:
                # extract out rows where Accession is in list
                data = data[data[index].isin(prot_list)]

            cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]
            
            if not cols:
                raise ValueError("No columns found. Please verify that the input 'cases' matches the columns in the data.")

            # concat elements 1 till end of vars into one string
            append_string = '_'.join(vars[1:])
            
            abundance = data[cols]
            abundance.index = data['Accession']

            # make dictionary for abundance and rank
            abun_dict[append_string] = abundance

        return abun_dict
    else:
        return {}

# TODO: maybe call stats_ttest instead
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

# TODO: fix to work with panndata
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

# !TODO: DONE W PANNDATA INTEGRATION | implement chi2 and fisher tests, consider also adding correlation tests
def get_DE(pdata, class_type, values, on = 'protein', method='ttest'):
    """
    Calculate differential expression (DE) of proteins across different groups.

    This function calculates the DE of proteins across different groups. The cases to compare can be specified, and the method to use for DE can be specified as well.

    Args:
        pdata (pAnnData): The pAnnData object containing the protein data.
        class_type (str): The class type to use for selecting samples. E.g. 'cell_type'.
        values (list of list of str): The values to select for within the class_type. E.g. [['wt', 'kd'], ['control', 'treatment']].
        on (str, optional): The type of data to perform DE on. Default is 'protein'. Other options include 'peptide'.
        method (str, optional): The method to use for DE. Default is 'ttest'. Other methods include 'mannwhitneyu', 'wilcoxon', 'chi2', and 'fisher'.

    Returns:
        df_stats (pandas.DataFrame): A DataFrame containing the DE statistics for each protein.

    Raises:
        ValueError: If the number of cases is not exactly two.

    Example:
        >>> from scviz import utils as scutils
        >>> stats_sc_20000 = scutils.get_DE(data, [['cortex','sc'], ['cortex','20000']])
    """

    # this is for case 1/case 2 comparison!
    # make sure only two cases are given
    if len(values) != 2:
        raise ValueError('Please provide exactly two cases to compare.')

    if on == 'protein':
        abundance_case1 = get_samples(pdata, class_type, values[0], exact_cases=True).prot
        abundance_case2 = get_samples(pdata, class_type, values[1], exact_cases=True).prot
    elif on == 'peptide':
        abundance_case1 = get_samples(pdata, class_type, values[0], exact_cases=True).pep
        abundance_case2 = get_samples(pdata, class_type, values[1], exact_cases=True).pep

    n1 = abundance_case1.shape[0]
    n2 = abundance_case2.shape[0]

    group1_string = '_'.join(values[0])
    group2_string = '_'.join(values[1])

    # create a dataframe for stats
    df_stats = pd.DataFrame(index=abundance_case1.var_names, columns=[group1_string,group2_string,'log2fc', 'p_value', 'test_statistic'])
    df_stats[group1_string] = np.mean(abundance_case1.X.toarray(), axis=0)
    df_stats[group2_string] = np.mean(abundance_case2.X.toarray(), axis=0)
    df_stats['log2fc'] = np.log2(np.divide(np.mean(abundance_case1.X.toarray(), axis=0), np.mean(abundance_case2.X.toarray(), axis=0)))

    if method == 'ttest':
        for protein in range(0, abundance_case1.shape[1]):
            t_test = ttest_ind(abundance_case1.X.toarray()[:,protein], abundance_case2.X.toarray()[:,protein])
            df_stats['p_value'].iloc[protein] = t_test.pvalue
            df_stats['test_statistic'].iloc[protein] = t_test.statistic
    elif method == 'mannwhitneyu':
        for row in range(0, len(abundance_case1)):
            mwu = mannwhitneyu(abundance_case1.iloc[row,0:n1-1].dropna().values, abundance_case2.iloc[row,0:n2-1].dropna().values)
            df_stats['p_value'].iloc[row] = mwu.pvalue
            df_stats['test_statistic'].iloc[row] = mwu.statistic
    elif method == 'wilcoxon':
        for row in range(0, len(abundance_case1)):
            w, p = wilcoxon(abundance_case1.iloc[row,0:n1-1].dropna().values, abundance_case2.iloc[row,0:n2-1].dropna().values)
            df_stats['p_value'].iloc[row] = p
            df_stats['test_statistic'].iloc[row] = w
    # elif method == 'chi2':
    #     for row in range(0, len(aligned_case1)):
    #         chi2 = chi2_contingency([aligned_case1.iloc[row,0:n1-1].dropna().values, aligned_case2.iloc[row,0:n2-1].dropna().values])
    #         df_stats['p_value'].iloc[row] = chi2.pvalue
    #         df_stats['t_statistic'].iloc[row] = chi2.statistic
    # elif method == 'fisher':
    #     for row in range(0, len(aligned_case1)):
    #         fisher = fisher_exact([aligned_case1.iloc[row,0:n1-1].dropna().values, aligned_case2.iloc[row,0:n2-1].dropna().values])
    #         df_stats['p_value'].iloc[row] = fisher[1]
    #         df_stats['t_statistic'].iloc[row] = fisher[0]

    return df_stats

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

def get_pca_importance(model, initial_feature_names):
    """
    Get the most important feature for each principal component in a PCA model.

    Args:
        model (sklearn.decomposition.PCA): The PCA model.
        initial_feature_names (list): The initial feature names. Typically adata.var_names.

    Returns:
        df (pd.DataFrame): A DataFrame containing the most important feature for each principal component.

    Example:
        >>> from scviz import utils as scutils
        >>> pca = PCA(n_components=5)
        >>> pca.fit(data)
        >>> df = scutils.get_pca_importance(pca)
    """

    # number of components
    n_pcs= model.components_.shape[0]

    # get the index of the most important feature on EACH component
    # LIST COMPREHENSION HERE
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

    # get the names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

    # LIST COMPREHENSION HERE AGAIN
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

    # build the dataframe
    df = pd.DataFrame(dic.items())

    return df

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


# NEED TO SOFTCODE THIS...
# def get_upset_contents(type):
#     ## start by making user specify the fixed variables and dependent variables (e.g. with assign values for fixed)
#     ## make use dictionary? e.g. fixed = {'grad_time': ['0', '1', '2'], 'region': ['cortex', 'snpc'], 'phenotype': ['sc', '4sc', '10c', '25c', '50c']}
#     ## if we can extract out all data for each fixed variable, then we can use the from_contents function to make the upset plot

#     # SPECIFY AMOUNTS
#     if type == 'amt':
#         # SPECIFY ENZYMES
#         amt_contents = {}
#         amts = ['sc', '4sc', '10c', '25c','50c']

#         # for same enzyme, compare amounts
#         for grad_time in grad_times:
#             amt_contents[grad_time] = {}
#             for region in regions:
#                 amt_contents[grad_time][region] = {}
#                 for phenotype in phenotypes:
#                     df_occ = pd.DataFrame(columns=['Protein', 'sc', '4sc', '10c', '25c', '50c', 'Total'])
#                     for amt in amts:
#                         cols = [col for col in data.columns if amt in col and phenotype in col and grad_time in col and region in col and 'Abundance: F' in col]
#                         df_occ[amt] = data[cols].notnull().sum(axis=1)
#                     df_occ['Protein'] = data['Accession']
#                     df_occ['Total'] = df_occ[amts].sum(axis=1)
#                     df_occ[amts] = df_occ[amts].astype(bool)
#                     content = {amt: df_occ['Protein'][df_occ[amt]].values for amt in amts}
#                     amt_contents[grad_time][region][phenotype] = from_contents(content)
        
#         return_contents = amt_contents

#     else:
#         print('Please specify either "amt" or "enzyme"')
#         return_contents = None

#     return return_contents

# TODO: add function to quickly drop/subset from data?

# TODO: add function to get GO enrichment, GSEA analysis (see GOATOOLS or STAGES or Enrichr?)
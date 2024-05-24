"""
This module contains functions for processing protein data and performing statistical tests.

Functions:
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

from scipy.sparse import csr_matrix
from sklearn.impute import SimpleImputer, KNNImputer
from scviz import pAnnData

# Thoughts: functions that act on panndata and return only panndata should be panndata methods, utility functions should be in utils

# TODO: move to class function  
def get_protein_summary(data, variables = ['region','amt']):
    """
    Import protein data from an Excel file and summarize characteristics about each sample and sample groups.

    This function reads an Excel file containing protein data, processes the data to extract relevant characteristics 
    for each sample, and summarizes the data for each sample group.

    Args:
        data (pandas.DataFrame): The protein data as a pandas DataFrame.
        variables (list): List of variables to extract from the column names. Default is ['region', 'amt'].

    Returns:
        df_files (pandas.DataFrame): A new DataFrame with the extracted data.
        protein_properties (pandas.DataFrame): A DataFrame containing the properties of each protein.


    Raises:
        None

    Example:
        >>> import scviz
        >>> summarized_data = scviz.data_utils.get_protein_summary(df, variables=['region', 'amt'])
    """

    df_prot_data = data.loc[~data['Description'].str.contains('CRAP')].copy()

    abundance_cols = [col for col in df_prot_data.columns if 'Abundance: ' in col]
    properties_endcol = df_prot_data.columns.get_loc("# Razor Peptides")
    protein_properties = df_prot_data.iloc[:, 1:properties_endcol] 

    # Extract the file name, and relevant sample typing from each column name
    file_names = [col.split(':')[1].strip() for col in abundance_cols]
    variables_list = [[col.split(':')[2].split(',')[i+1].strip() for col in abundance_cols] for i in range(len(variables))]

    df_files = pd.DataFrame({'file_name': file_names, **{variables[i]: variables_list[i] for i in range(len(variables))}})

    for f in df_files.file_name:
        # Extract abundance columsn from data that has the file name in its column name
        df = df_prot_data[[col for col in df_prot_data.columns if f+":" in col]]
        
        # if df.column[1] doesn't include found, swap columns 0 and 1
        if 'Found' not in df.columns[1]:
            df = df.iloc[:, [1, 0]]
        
        # Count the number of rows that meet each condition
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')

        cond1_count = df[(df.iloc[:, 0].notnull()) & (df.iloc[:, 1] == 'High')].shape[0] # quantified and identified
        cond2_count = df[(df.iloc[:, 0].isnull()) & (df.iloc[:, 1] == 'High')].shape[0] # not quantified but identified
        cond3_count = df[(df.iloc[:, 0].notnull()) & (df.iloc[:, 1] == 'Peak Found')].shape[0] # quantified and MBR
        cond4_count = df[(df.iloc[:, 0].isnull()) & (df.iloc[:, 1] == 'Peak Found')].shape[0] # not quantified but MBR

        # Add df_quant to gluc_files on the rows where f matches file_names
        df_quant = (cond1_count + cond3_count) / (cond1_count + cond2_count + cond3_count + cond4_count)
        df_files.loc[df_files['file_name'] == f, 'df_quant'] = df_quant

        # count the number of high confidence and peak found (mbr) proteins
        high_count = df[df.iloc[:, 1] == 'High'].shape[0]
        mbr_count = df[df.iloc[:, 1] == 'Peak Found'].shape[0]
        # count the number of high confidence and peak found proteins with >1 unique peptides (10th column of protein_properties)
        pep1_count = protein_properties[(df.iloc[:, 1] == 'High') | (df.iloc[:, 1] == 'Peak Found') & (protein_properties.iloc[:, 9] > 1)].shape[0]
        pep2_count = protein_properties[(df.iloc[:, 1] == 'High') | (df.iloc[:, 1] == 'Peak Found') & (protein_properties.iloc[:, 9] > 2)].shape[0]
        pep_count = protein_properties[(df.iloc[:, 1] == 'High') | (df.iloc[:, 1] == 'Peak Found')].iloc[:, 5].sum()

        df_files.loc[df_files['file_name'] == f, ['high_count', 'mbr_count', 'total_count', 'pep1_count', 'pep2_count', 'pep_count']] = [high_count, mbr_count, high_count + mbr_count, pep1_count, pep2_count, pep_count]        

    # assign replicate number column
    df_files['replicate'] = df_files.groupby(variables).cumcount() + 1

    df_files = df_files[['file_name'] + variables + ['replicate', 'df_quant', 'high_count', 'mbr_count', 'total_count', 'pep1_count', 'pep2_count', 'pep_count']]

    return df_files, protein_properties

# TODO: move to class function | fix to do own normalization
def get_protein_norm(data, norm_data_fp, norm_list_fp, norm_type = 'auto', export=False):
    """
    Append normalized protein data to the original protein data.

    This function reads a protein data file and a normalization data file (with its input sample list), and appends the normalized data to the original data.

    Args:
        data (pandas.DataFrame): The original protein data as a pandas DataFrame.
        norm_data_fp (str): The file path to the normalization data.
        norm_list_fp (str): The file path to the normalization sample list.
        norm_type (str): The type of normalization to append. Default is 'auto'. If 'auto', the function will use the first normalization type found in the normalization data file.
        export (bool): Whether to export the appended data to a new file. Default is False.

    Returns:
        pandas.DataFrame: A new DataFrame with the appended normalized data.

    Raises:
        None

    Example:
        >>> import scviz
        >>> data_norm = scviz.utils.get_protein_norm(data, norm_data_fp, norm_list_fp, norm_type='linear', export=True)
    """
    
    norm_list = pd.read_csv(norm_list_fp)
    norm_data = pd.read_csv(norm_data_fp)
    append_data = data.copy()

    if norm_type == 'auto':
        # if the norm type is not specified, then use the first norm type (after raw) found in the norm_data.columns
        # skip all columns that contain 'abundance_raw' in the column name
        norm_type = [col for col in norm_data.columns if 'abundance_' in col and 'raw' not in col][0].split('_')[1]
        print("Normalization type not specified, "+norm_type+" type normalization data found in "+norm_data_fp)
    else:
        if not(any(norm_type in col for col in norm_data.columns)):
            raise ValueError(f"Norm type {norm_type} not found in {norm_data_fp}")

    print("Processing "+norm_type+" type normalization data found in "+norm_data_fp)

    # from svm_list, make a dictionary where sample+"_"+replicate is the key, and sample_file is the value
    norm_dict = {}
    for i in range(len(norm_list)):
        norm_dict[str(norm_list.iloc[i, 4]) + "_" + str(norm_list.iloc[i, 5])] = norm_list.iloc[i, 3]

    # appending norm data to original data by the following steps
    # 1. In data, for each column of "Abundance: F(number): etc", create another column of "(norm) Abundance: F(number): etc"
    abundance_cols = [col for col in append_data.columns if 'Abundance: F' in col]

    # 2. For each row in norm_data, find the corresponding row in data, and copy the norm abundance data to the new column
    for i in range(norm_data.shape[0]):
        # find row where accession in data matches to protein in svm_data
        row = append_data[append_data['Accession'] == norm_data.iloc[i, 0]]
        norm_cols = [col for col in norm_data.columns if 'abundance_'+norm_type in col]

        # for each column, extract the sample information that appears after abundance_norm_sampleinformation
        for col in norm_cols:
            sample_info = col.split('abundance_'+norm_type+"_")[1]
            # if the sample info is in the norm_dict, then copy the norm abundance to the corresponding column in data
            if sample_info in norm_dict:
                norm_abundance = norm_data.iloc[i, norm_data.columns.get_loc(col)]
                info = 'Sample, ' + ', '.join(sample_info.split('_'))
                # capitalize norm_type
                append_data.loc[row.index, norm_type.capitalize()+' Abundance: ' + norm_dict[sample_info] + ': ' + info] = norm_abundance

    # remove columns of raw abundance i.e. they start with "Abudance: F"
    append_data = append_data.drop(abundance_cols, axis=1)

    # export the data to a new excel file
    if export:
        append_data.to_csv(norm_data_fp.split('.')[0]+'_norm_'+norm_type+'.csv', index=False)

    return append_data

# !TODO: for peptide, use 'Annotated Sequence' instead of 'Accession' and shared search subset='Annotated Sequence'
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

# TODO: move to class function | fix now w pAnnData    
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
    
# TODO: move to class function | just use anndata splicing?
def filter_by_group(df, variables, values):
    """
    Filter a DataFrame based on specified groups. Helper function for run_ttest.

    Args:
        df (pandas.DataFrame): The DataFrame to filter.
        variables (list): The variables to use for grouping.
        values (list): The values to use for filtering.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    return df[np.all([df[variables[i]] == values[i] for i in range(len(variables))], axis=0)]

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

def get_abundance_query(data, cases, genelist, search='gene'):
    """
    Search and extract protein abundance data based on a specific gene list.

    This function searches and extracts protein abundance data for specified cases based on a specific gene list. The search can be performed on 'gene', 'protein', 'description', 'pathway', or 'all'. It also accepts a list of terms.

    Args:
        data (pandas.DataFrame): The protein abundance data.
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

# !TODO: implement chi2 and fisher tests, consider also adding correlation tests
def get_protein_DE(data, cases, method='ttest'):
    """
    Calculate differential expression (DE) of proteins across different groups.

    This function calculates the DE of proteins across different groups. The cases to compare can be specified, and the method to use for DE can be specified as well.

    Args:
        data (pandas.DataFrame): The protein data.
        cases (list): The cases to compare.
        method (str, optional): The method to use for DE. Default is 'ttest'. Other methods include 'mannwhitneyu', 'wilcoxon', 'chi2', and 'fisher'.

    Returns:
        df_stats (pandas.DataFrame): A DataFrame containing the DE statistics for each protein.

    Raises:
        ValueError: If the number of cases is not exactly two.

    Example:
        >>> from scviz import utils as scutils
        >>> stats_sc_20000 = scutils.get_protein_DE(data, [['cortex','sc'], ['cortex','20000']])
    """

    # this is for case 1/case 2 comparison!
    # make sure only two cases are given
    if len(cases) != 2:
        raise ValueError('Please provide exactly two cases to compare.')

    dict_cases_raw = get_abundance(data, cases, abun_type='raw')

    n1 = dict_cases_raw[list(dict_cases_raw.keys())[0]].shape[1]
    n2 = dict_cases_raw[list(dict_cases_raw.keys())[1]].shape[1]

    aligned_case1, aligned_case2 = dict_cases_raw[list(dict_cases_raw.keys())[0]].align(dict_cases_raw[list(dict_cases_raw.keys())[1]])

    group1_string = '_'.join(cases[0])
    group2_string = '_'.join(cases[1])

    # create a dataframe for stats
    df_stats = pd.DataFrame(index=aligned_case1.index, columns=[group1_string,group2_string,'log2fc', 'p_value', 'test_statistic'])
    df_stats[group1_string] = aligned_case1.mean(axis=1)
    df_stats[group2_string] = aligned_case2.mean(axis=1)
    df_stats['log2fc'] = np.log2(np.divide(aligned_case1.mean(axis=1), aligned_case2.mean(axis=1)))

    if method == 'ttest':
        for row in range(0, len(aligned_case1)):
            t_test = ttest_ind(aligned_case1.iloc[row,0:n1-1].dropna().values, aligned_case2.iloc[row,0:n2-1].dropna().values)
            df_stats['p_value'].iloc[row] = t_test.pvalue
            df_stats['test_statistic'].iloc[row] = t_test.statistic
    elif method == 'mannwhitneyu':
        for row in range(0, len(aligned_case1)):
            mwu = mannwhitneyu(aligned_case1.iloc[row,0:n1-1].dropna().values, aligned_case2.iloc[row,0:n2-1].dropna().values)
            df_stats['p_value'].iloc[row] = mwu.pvalue
            df_stats['test_statistic'].iloc[row] = mwu.statistic
    elif method == 'wilcoxon':
        for row in range(0, len(aligned_case1)):
            w, p = wilcoxon(aligned_case1.iloc[row,0:n1-1].dropna().values, aligned_case2.iloc[row,0:n2-1].dropna().values)
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
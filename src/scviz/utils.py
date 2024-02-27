"""
This module contains functions for importing and manipulating data.

Functions:
    function1: This function is responsible for...
    function2: This function is responsible for...
    ...

Note:
    Replace 'function1', 'function2', etc. with your actual function names and provide a brief description of what each function does.

Example:
    To use this module, import it and call functions from your code as follows:

    import data_processing
    processed_data = data_processing.function1(raw_data)

Todo:
    * For future implementation.
"""
import pandas as pd
import numpy as np
import re

from scipy.stats import ttest_ind
from decimal import Decimal

    
def protein_summary(data, variables = ['region','amt']):
    """
    Import protein data from an Excel file and summarize characteristics about each sample and sample groups.

    This function reads an Excel file containing protein data, processes the data to extract relevant characteristics 
    for each sample, and summarizes the data for each sample group.

    Args:
        data (pandas.DataFrame): The protein data as a pandas DataFrame.
        variables (list): List of variables to extract from the column names. Default is ['region', 'amt'].

    Returns:
        pandas.DataFrame: A new DataFrame with the extracted data.

    Raises:
        None

    Example:
        >>> import scviz
        >>> summarized_data = scviz.data_utils.protein_summary(df, variables=['region', 'amt'])
    """

    df_prot_data = data.loc[~data['Description'].str.contains('CRAP')].copy()

    abundance_cols = [col for col in df_prot_data.columns if 'Abundance: ' in col]
    properties_endcol = df_prot_data.columns.get_loc("# Razor Peptides")
    protein_properties = df_prot_data.iloc[:, 1:properties_endcol]
    # note that accession properties (biological processes, cellular components, molecular function) are columns ?-? respectively

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

    return df_files

def append_norm(data, norm_data_fp, norm_list_fp, norm_type = 'auto', export=False):
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
        >>> data_norm = scviz.utils.append_norm(data, norm_data_fp, norm_list_fp, norm_type='linear', export=True)
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
            print("Norm type "+norm_type+" not found in "+norm_data_fp)
            return

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

def get_cv(data, cases, variables=['region', 'amt'], sharedPeptides = False):
    """
    Calculate the coefficient of variation (CV) for each case in the given data.

    Parameters:
    - data: pandas DataFrame
        The input data containing the CV values.
    - cases: list of lists
        The cases to calculate CV for. Each case is a list of values corresponding to the variables.
    - variables: list, optional
        The variables to consider when calculating CV. Default is ['region', 'amt'].
    - sharedPeptides: bool, optional
        Whether to calculate CV for only shared peptides identified across all cases. Default is False.

    Returns:
    - cv_df: pandas DataFrame
        The DataFrame containing the CV values for each case, along with the corresponding variable values.
    """
    
    # check if the len of each element in cases have the same length as len(variables), else throw error message
    if not all(len(cases[i]) == len(variables) for i in range(len(cases))):
        print("Error: length of each element in cases must be equal to length of variables")
        return
    
    # make dataframe for each case with all CV values
    cv_df = pd.DataFrame()
    data = data.copy()

    if sharedPeptides:
        all_cvs = []
        for j in range(len(cases)):
            vars = ['CV'] + cases[j]
            cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]
            for i in range(len(cols)):
                all_cvs.append(data[cols[i]].values)

        # find the rows that contain one or more NaNs
        nan_rows = data[data.isnull().any(axis=1)]
        # remove the rows that contain one or more NaNs
        data = data.drop(nan_rows.index)

    for j in range(len(cases)):
        vars = ['CV'] + cases[j]
        cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]
        nsample = len(cols)

        print(vars)
        print(cols)

        # merge all CV columns into one column
        X = np.zeros((nsample*len(data)))
        for i in range(nsample):
            X[i*len(data):(i+1)*len(data)] = data[cols[i]].values
        # remove nans
        X = X[~np.isnan(X)]/100

        # add X to cur_df, and append case info of enzyme, method and amt to each row
        cur_df = pd.DataFrame()
        cur_df['cv'] = X
        for i in range(len(variables)):
            cur_df[variables[i]] = cases[j][i]
        
        print(cur_df)

        # append cur_df to cv_df
        cv_df = pd.concat([cv_df, cur_df], ignore_index=True)

    return cv_df

def return_abundance(data,cases,names=None, abun_type='average', num_cat=2):
    if abun_type=='average':
        # create empty list to store abundance values
        abun_dict = {}
        data = data.copy()
        # extract columns that contain the abundance data for the specified method and amount
        for j in range(len(cases)):
            vars = ['Abundance: '] + cases[j]

            if names is not None:
                # extract out rows where Accession is in names
                data = data[data['Accession'].isin(names)]

            cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]
            # concat elements 1 till end of vars into one string
            append_string = '_'.join(vars[1:])
            
            # average abundance of proteins across these columns, ignoring NaN values
            data['Average: '+append_string] = data[cols].mean(axis=1, skipna=True)
            data['Stdev: '+append_string] = data[cols].std(axis=1, skipna=True)

            # sort by average abundance
            data.sort_values(by=['Average: '+append_string], ascending=False, inplace=True)
            abundance=data['Average: '+append_string]
            accession=data['Accession']

            # add rank number
            rank=data['Rank: '+append_string] = np.arange(1, len(data)+1)

            # make dictionary for abundance and rank
            abun_dict[append_string] = [abundance, rank, accession]
        return abun_dict

    if abun_type == 'raw':
        # create empty list to store abundance values
        abun_dict = {}
        data = data.copy()
        # extract columns that contain the abundance data for the specified method and amount
        for j in range(len(cases)):
            vars = ['Abundance: '] + cases[j]

            if names is not None:
                # extract out rows where Accession is in names
                data = data[data['Accession'].isin(names)]

            cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]
            # concat elements 1 till end of vars into one string
            append_string = '_'.join(vars[1:])
            
            abundance = data[cols]
            accession=data['Accession']

            # make dictionary for abundance and rank
            abun_dict[append_string] = [abundance, accession]

        return abun_dict
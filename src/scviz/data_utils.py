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
    
def protein_data_summary(file_path, variables = ['region','amt']):
    """
    Import protein data from an Excel file and summarize characteristics about each sample and sample groups.

    This function reads an Excel file containing protein data, processes the data to extract relevant characteristics 
    for each sample, and summarizes the data for each sample group.

    Args:
        df_prot_data (pandas.DataFrame): The protein data as a pandas DataFrame.
        variables (list): List of variables to extract from the column names. Default is ['region', 'amt'].

    Returns:
        pandas.DataFrame: A new DataFrame with the extracted data.

    Raises:
        None

    Example:
        >>> import scviz
        >>> summarized_data = scviz.data_utils.protein_data_summary(df, variables=['region', 'amt'])
    """

    df_prot_data = pd.read_excel(file_path)
    df_prot_data = df_prot_data[~df_prot_data['Description'].str.contains('CRAP')]

    abundance_cols = [col for col in df_prot_data.columns if 'Abundance: ' in col]
    properties_endcol = df_prot_data.columns.get_loc("# Razor Peptides")
    protein_properties = df_prot_data.iloc[:, 1:properties_endcol]
    # note that accession properties (biological processes, cellular components, molecular function) are columns ?-? respectively

    # Extract the file name, and relevant sample typing from each column name
    file_names = [col.split(':')[1].strip() for col in abundance_cols]
    variables_list = []
    variables_list = [[col.split(':')[2].split(',')[i+1].strip() for col in abundance_cols] for i in range(len(variables))]

    df_files = pd.DataFrame({'file_name': file_names, **{variables[i]: variables_list[i] for i in range(len(variables))}})

    for f in df_files.file_name:
        # Extract any column from data that has the file name in its column name
        df_cols = [col for col in df_prot_data.columns if f+":" in col]
        # extract the columns that contain the abundance and found in data
        df = df_prot_data[df_cols]

        # if df.column[1] doesn't include found, swap columns 0 and 1
        if 'Found' not in df.columns[1]:
            df = df.iloc[:, [1, 0]]
        
        # Count the number of rows that meet each condition
        cond1_count = df[(pd.to_numeric(df.iloc[:, 0], errors='coerce').notnull()) & (df.iloc[:, 1] == 'High')].shape[0]
        cond2_count = df[(pd.to_numeric(df.iloc[:, 0], errors='coerce').isnull()) & (df.iloc[:, 1] == 'High')].shape[0]
        cond3_count = df[(pd.to_numeric(df.iloc[:, 0], errors='coerce').notnull()) & (df.iloc[:, 1] == 'Peak Found')].shape[0]
        cond4_count = df[(pd.to_numeric(df.iloc[:, 0], errors='coerce').isnull()) & (df.iloc[:, 1] == 'Peak Found')].shape[0]
        cond5_count = df[df.iloc[:,1] == 'Not Found'].shape[0]

        # Add df_quant to gluc_files on the rows where f matches file_names
        df_quant = (cond1_count + cond3_count) / (cond1_count + cond2_count + cond3_count + cond4_count)
        df_files.loc[df_files['file_name'] == f, 'df_quant'] = df_quant

        # count the number of high confidence and peak found (mbr) proteins
        high_count = df[df.iloc[:, 1] == 'High'].shape[0]
        mbr_count = df[df.iloc[:, 1] == 'Peak Found'].shape[0]
        # count the number of high confidence and peak found proteins with >1 unique peptides (10th column of protein_properties)
        total_count_1pep = protein_properties[(df.iloc[:, 1] == 'High') | (df.iloc[:, 1] == 'Peak Found') & (protein_properties.iloc[:, 9] > 1)].shape[0]
        total_count_2pep = protein_properties[(df.iloc[:, 1] == 'High') | (df.iloc[:, 1] == 'Peak Found') & (protein_properties.iloc[:, 9] > 2)].shape[0]

        # add high and mbr counts to df_files, and total_count_1pep
        df_files.loc[df_files['file_name'] == f, 'high_count'] = high_count
        df_files.loc[df_files['file_name'] == f, 'mbr_count'] = mbr_count
        # sum high_count and mbr_count as total_count, and add to df_file
        df_files.loc[df_files['file_name'] == f, 'total_count'] = high_count + mbr_count
        df_files.loc[df_files['file_name'] == f, 'total_count_1pep'] = total_count_1pep
        df_files.loc[df_files['file_name'] == f, 'total_count_2pep'] = total_count_2pep
        
        # total pep count is the sum of peptide number (sixth column of protein_properties) for rows where the protein was found in the sample (where second column of df is high or peak found)
        total_pep_count = protein_properties[(df.iloc[:, 1] == 'High') | (df.iloc[:, 1] == 'Peak Found')].iloc[:, 5].sum()
        # add total pep count to df_files
        df_files.loc[df_files['file_name'] == f, 'total_pep_count'] = total_pep_count

    # assign replicate number column
    df_files['replicate'] = 0
    for i in range(df_files.shape[0]):
        # find rows where region, amt, phenotype, organism and grad_time match
        rows = df_files[(df_files['region'] == df_files.iloc[i, 1]) & (df_files['amt'] == df_files.iloc[i, 2])]
        # for each row, assign replicate number starting from 1
        for j in range(rows.shape[0]):
            df_files.loc[rows.index[j], 'replicate'] = j+1

    # move replicate column to the front
    df_files_column = ['file_name']
    df_files_column.extend(variables)
    df_files_column.extend(['replicate', 'df_quant', 'high_count', 'mbr_count', 'total_count', 'total_count_1pep', 'total_count_2pep', 'total_pep_count'])

    df_files = df_files[df_files_column]

    return df_files

def append_prot_norm(data, norm_data_fp, norm_list_fp, norm_type = 'linear', export=False):
    """
    Append normalized protein data to the original protein data.

    This function reads a protein data file and a normalization data file, and appends the normalized data to the original data.

    Args:
        data (pandas.DataFrame): The original protein data as a pandas DataFrame.
        norm_data_fp (str): The file path to the normalization data.
        norm_list_fp (str): The file path to the normalization list.
        norm_type (str): The type of normalization to append. Default is 'linear'.
        export (bool): Whether to export the appended data to a new file. Default is False.

    Returns:
        pandas.DataFrame: A new DataFrame with the appended normalized data.

    Raises:
        None

    Example:
        >>> import scviz
        >>> appended_data = scviz.data_utils.append_prot_norm(data, norm_data_fp, norm_list_fp, norm_type='linear', export=True)
    """
    
    norm_list = pd.read_csv(norm_list_fp)
    norm_data = pd.read_csv(norm_data_fp)
    append_data = data.copy()

    if not(any(norm_type in col for col in norm_data.columns)):
        print("Norm type "+norm_type+" not found in "+norm_data_fp)
        return

    print(norm_type+" found in "+norm_data_fp)

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
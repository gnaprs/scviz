def get_unique_proteins_from_terms(df, terms, label_col="matching proteins in your network (labels)", term_col="term description"):
    """
    Return the unique set of proteins from the 'labels' column
    for the given list of term descriptions.

    tsv of results should be downloaded from string website

    Parameters:
        df (pd.DataFrame): The enrichment results dataframe.
        terms (list of str): List of term descriptions to match.
        label_col (str): Column containing protein labels (comma-separated).
        term_col (str): Column containing the GO/KEGG/etc term descriptions.

    Returns:
        Set of unique protein labels across all matching terms.
    """
    matched_rows = df[df[term_col].isin(terms)]
    protein_lists = matched_rows[label_col].dropna().str.split(',')
    flat_list = [protein.strip() for sublist in protein_lists for protein in sublist]
    return list(set(flat_list))


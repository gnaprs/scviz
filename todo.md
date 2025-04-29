## To-Do List

# TODO features to implement
0. Consider adding function that tidies up anndata.X -> so that we can directly use any sc.tl/sc.pl function
1. Add sharedPeptides function on get_CV()
2. Update umap with uncertainty and also subset coloring
3. Implement dot plot: expression of the indicated genes in the three clusters (dot size is percentage of cells per cluster; color is cluster average normalized expression)
4. implement directLFQ () for normalization, double check normalization (median?) from 240731_cbr
5. implement correlation visualization (protein corr module)
6. implement peptide sequence characteristics (hydrophobicity, etc.)
7. implement imputation within classes (nested), not sure if can use ColumnTransformer together with SimpleImputer for this (check dev_imputation.ipynb)

# TESTS
1. import tests not yet done - have test files for pd and diann already
2. filter tests need to be checked
3. implement quick utils check? 

# FIX/BUGS
1. sync get_uniprot_fields() with convert_identifiers() and get_string_... functions - make one combined function for all 3
2. housekeeping csvs not installed with package - find a way to fix this
3. get_pca_importance just needs to accept pdata input (and prot/pep) - can get uns['pca'] and var_names from it.
4. plot_abundance stopped showing data points? check?

# MAINTENANCE
1. Check out scprep repo for possible utility functions
2. Add Type hints to all functions, try to fix all the lint errors...
3. Double-check peptide export format: (i) Gene Name (ii) Peptide name (iii) Peptide amino acid sequence start and end in the protein (iv) Charge, etc. if any (v) and then columns with the sample names where the intensities are reported

# DOCUMENTATION
1. yes.
2. But also a tutorial ipynb
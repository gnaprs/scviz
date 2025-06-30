## To-Do List

# TODO features to implement
0. implement harmony
1. Add sharedPeptides function on get_CV()
2. Update umap with uncertainty and also subset coloring
3. Implement dot plot: expression of the indicated genes in the three clusters (dot size is percentage of cells per cluster; color is cluster average normalized expression)
4. implement directLFQ() for normalization
5. implement correlation visualization (protein corr module)
6. implement peptide sequence characteristics (hydrophobicity, etc.)
7. check for DIA/DDA - and suggest preprocessing methods for these?
8. concating multiple datasets? - see ad.concat, maybe use adata.obs_names_make_unique()
9. consider QC metrics beyond what we already have?
10. add gene to uniprot accession conversion function
11. allow users to input "obs" for their own metadata
12. string db values rank api ([text](https://string-db.org/cgi/help?subpage=api%23valuesranks-enrichment-api)) - also check about string API key; maybe can generate one per user instead? (store local copy in package if that's possible)

# ENHANCEMENTS (not as important)
1. when updating .summary/obs, move prot/pep details to the right (prioritize metadata)
2. heatmap implemented, but may be better to just export to a format for morpheous

# TESTS
1. reorganize tests per suite in panndata module
2. Check clean_X and verify works with scanpy (tidies up anndata.X -> so that we can directly use any sc.tl/sc.pl function)

# FIX/BUGS
1. sync get_uniprot_fields() with convert_identifiers() and get_string_... functions - make one combined function for all 3
2. housekeeping csvs not installed with package - find a way to fix this
3. get_pca_importance just needs to accept pdata input (and prot/pep) - can get uns['pca'] and var_names from it.
4. plot_pca ellipse having issues (only showing one ellipse, not around data points)
5. new pd3.1 has new format for peptides - modifications column name definitely changed, and i think uniprot gene/prot checking may be bugged
6. .summary updates number of proteins, but repr(pdata) doesn't
7. move normalize USER print statements to before error messages

# MAINTENANCE
1. Check out scprep repo for possible utility functions
2. Python linting errors (github)
3. Double-check peptide export format: (i) Gene Name (ii) Peptide name (iii) Peptide amino acid sequence start and end in the protein (iv) Charge, etc. if any (v) and then columns with the sample names where the intensities are reported

# DOCUMENTATION
1. Unify all verbose messages - also decide on what to show normally, what to show if debug
2. But also a tutorial ipynb
## To-Do List

# TODO features to implement
0. Consider adding function that tidies up anndata.X -> so that we can directly use any sc.tl/sc.pl function
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

# ENHANCEMENTS (not as important)
1. when updating .summary/obs, move prot/pep details to the right (prioritize metadata)

# TESTS
1.  

# FIX/BUGS
1. sync get_uniprot_fields() with convert_identifiers() and get_string_... functions - make one combined function for all 3
2. housekeeping csvs not installed with package - find a way to fix this
3. get_pca_importance just needs to accept pdata input (and prot/pep) - can get uns['pca'] and var_names from it.
4. plot_pca ellipse having issues (only showing one ellipse, not around data points)
5. new pd3.1 has new format for peptides - modifications dolumn name definitely changed, and i think uniprot gene/prot checking may be bugged

# MAINTENANCE
1. Check out scprep repo for possible utility functions
2. Python linting errors (github)
3. Double-check peptide export format: (i) Gene Name (ii) Peptide name (iii) Peptide amino acid sequence start and end in the protein (iv) Charge, etc. if any (v) and then columns with the sample names where the intensities are reported

# DOCUMENTATION
1. Unify all verbose messages - also decide on what to show normally, what to show if debug
2. But also a tutorial ipynb
## To-Do List

# TODO features to implement
0. allow users to input "obs" for their own metadata
1. Add sharedPeptides function on get_CV()
2. Implement dot plot: expression of the indicated genes in the three clusters (dot size is percentage of cells per cluster; color is cluster average normalized expression)
3. implement directLFQ() for normalization
4. implement correlation visualization (protein corr module)
5. implement peptide sequence characteristics (hydrophobicity, etc.)
7. concating multiple datasets? - see ad.concat, maybe use adata.obs_names_make_unique()
8. consider QC metrics beyond what we already have?
9. add gene to uniprot accession conversion function
10. string db values rank api ([text](https://string-db.org/cgi/help?subpage=api%23valuesranks-enrichment-api)) - also check about string API key; maybe can generate one per user instead? (store local copy in package if that's possible)
11. Add tools for comparing two comparisons - plot of log(fc) vs log (fc), with coloring for significance? | then we also need to control hypothesis testing - ANOVA with Tukey and BH-algorithm for correction [see tidyproteomics maybe]

# TODO (non-features)
1. Tutorial to show how to integrate with scanpy features
2. Clean up readme to point to API and tutorials

# ENHANCEMENTS (not as important)
1. when updating .summary/obs, move prot/pep details to the right (prioritize metadata)
2. let user pass on their own transformer for pdata.neighbor
3. implement string without mapping? mapping is very slow...
4. get_pca_importance just needs to accept pdata input (and prot/pep) - can get uns['pca'] and var_names from it.
5. Double-check peptide export format: (i) Gene Name (ii) Peptide name (iii) Peptide amino acid sequence start and end in the protein (iv) Charge, etc. if any (v) and then columns with the sample names where the intensities are reported
6. check for DIA/DDA - and suggest preprocessing methods for these?
7. fuzzy match for get_abundance matches
8. Move Found In and Significant In columns from var/obs...? Too many columns... Or more concisely express these

# TESTS
1. reorganize tests per suite in panndata module
2. Check clean_X and verify works with scanpy (tidies up anndata.X -> so that we can directly use any sc.tl/sc.pl function)

# FIX/BUGS
1. sync get_uniprot_fields() with convert_identifiers() and get_string_... functions - make one combined function for all 3
2. housekeeping csvs not installed with package - find a way to fix this
3. new pd3.1 has new format for peptides - modifications column name definitely changed, and i think uniprot gene/prot checking may be bugged
4. .summary updates number of proteins, but repr(pdata) doesn't
5. clustermap bug when linkage
6. string error for mappings? Also double check the way I check species

# MAINTENANCE
1. Check out scprep repo for possible utility functions
2. Python linting errors (github)

# DOCUMENTATION
1. Unify all verbose messages - also decide on what to show normally, what to show if debug
2. But also a tutorial ipynb
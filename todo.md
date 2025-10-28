# TODO 
## FEATURES
### base
1. allow users to input "obs" for their own metadata
2. Add sharedPeptides function on get_CV()
3. concating multiple datasets? - see ad.concat, maybe use adata.obs_names_make_unique()
4. add gene to uniprot accession conversion function
### statistics/data viz
5. Implement dot plot: expression of the indicated genes in the three clusters (dot size is percentage of cells per cluster; color is cluster average normalized expression)
6. Add tools for comparing two comparisons - plot of log(fc) vs log (fc), with coloring for significance? | then we also need to control hypothesis testing - ANOVA with Tukey and BH-algorithm for correction [see tidyproteomics maybe]
### enrichment
7. string db values rank api ([text](https://string-db.org/cgi/help?subpage=api%23valuesranks-enrichment-api)) - also check about string API key; maybe can generate one per user instead? (store local copy in package if that's possible)
### new modules
8. implement peptide sequence characteristics (hydrophobicity, etc.) [peptide_param module]
9. implement correlation visualization (protein corr module)
### others
10. consider QC metrics beyond what we already have?

## ENHANCEMENTS
1. sync get_uniprot_fields() with convert_identifiers() and get_string_... functions - make one combined function for all 3
1. when updating .summary/obs, move prot/pep details to the right (prioritize metadata)
2. let user pass on their own transformer for pdata.neighbor
3. get_pca_importance just needs to accept pdata input (and prot/pep) - can get uns['pca'] and var_names from it.
4. Double-check peptide export format: (i) Gene Name (ii) Peptide name (iii) Peptide amino acid sequence start and end in the protein (iv) Charge, etc. if any (v) and then columns with the sample names where the intensities are reported
5. check for DIA/DDA - and suggest preprocessing methods for these?
6. fuzzy match for get_abundance matches
7. Move Found In and Significant In columns from var/obs...? Too many columns... Or more concisely express these
8. filter sample filelist to accept NOT file_list instead (maybe remove=True flag?)

# TESTS
## KNOWN FIX/BUGS
2. housekeeping csvs not installed with package - find a way to fix this
3. new pd3.1 has new format for peptides - modifications column name definitely changed, and i think uniprot gene/prot checking may be bugged
4. .summary updates number of proteins, but repr(pdata) doesn't
5. clustermap bug when linkage
6. plot_abundnace_housekeeping throws error when no housekeeping gene is found
7. check that we're syncing rs and filtering rs matrix for every filter opration (currently, only for filter sample by condition?)
9. When filtering by samples, need to clean up empty proteins?
10. when impute throws error because wrong obs column given, pretty format the error so that people understand better

## MAINTENANCE
1. Check out scprep repo for possible utility functions
2. Python linting errors (github)

## DOCUMENTATION
1. Tutorial to show how to integrate with scanpy features
2. QC tutorial
3. search_annotation tutorial
## To-Do List
1. Add more comprehensive documentation.

# TODO features to implement
1. Add sharedPeptides function on get_CV()
2. Protein functionality - how many peptides/unique peptides per protein (+ TRUE/FALSE column for >=1 peptide common between samples with that peptide)
3. Double-check peptide export format: (i) Gene Name (ii) Peptide name (iii) Peptide amino acid sequence start and end in the protein (iv) Charge, etc. if any (v) and then columns with the sample names where the intensities are reported
4. Update umap with uncertainty and also subset coloring
Implement dot plot: expression of the indicated genes in the three clusters (dot size is percentage of cells per cluster; color is cluster average normalized expression)
5. implement directLFQ () for normalization
6. implement correlation visualization

# FIX/BUGS
1. get_query()
2. get_abundance_query()
3. sync get_uniprot_fields() with convert_identifiers()
4. sync get_uniprot_fields() with get_string_... functions - make one combined function for all 3

# MAINTENANCE
1. Tests for imports - new+old DIANN format, new PD DIA format
2. Check out scprep repo for possible utility functions
3. Add Type hints to all functions, e.g. 
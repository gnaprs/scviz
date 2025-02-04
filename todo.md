## To-Do List
1. Add more visualization features (correlation matrix)
2. Extend support for GSEA and GO analysis
3. Add more comprehensive documentation.
4. Implement unit tests for better code reliability.

# TODO For marion
Add sharedPeptides function on get_CV()
Check out scprep repo for possible utility functions
Protein functionality - how many peptides/unique peptides per protein (+ TRUE/FALSE column for >=1 peptide common between samples with that peptide)
Double-check peptide export format: (i) Gene Name (ii) Peptide name (iii) Peptide amino acid sequence start and end in the protein (iv) Charge, etc. if any (v) and then columns with the sample names where the intensities are reported
Add Type hints to all functions, e.g. 
Update umap with uncertainty and also subset coloring
Implement dot plot: expression of the indicated genes in the three clusters (dot size is percentage of cells per cluster; color is cluster average normalized expression)

# CHECK
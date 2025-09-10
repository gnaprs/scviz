# Welcome to scviz
**scviz** is a Python package for single-cell and spatial proteomics data analysis, built around a custom `pAnnData` object.  
It extends the [AnnData](https://anndata.readthedocs.io/) ecosystem with proteomics-specific functionality, enabling seamless integration of proteins, peptides, and relational data.
## Features
- **Single-cell proteomics support**: Store protein and peptide quantifications in AnnData-compatible structures.  
- **Relational mapping**: Track protein–peptide connectivity using a dedicated RS matrix.  
- **Analysis tools**: Filtering, normalization, imputation, and differential expression (DE) tailored for proteomics.  
- **Functional enrichment**: Integrated [STRING](https://string-db.org/) queries for GSEA and PPI networks.  
- **Custom plotting**: Publication-ready plots (abundance, PCA/UMAP, clustermaps, rank-quant plots, etc.).  
- **API utilities**: Retrieve annotations from UniProt, cache mappings, and manage large datasets efficiently.  
## Getting Started
Install the latest version of scviz:

```bash
pip install scviz
```

Import and create a `pAnnData` object:

```python
import scviz as scv

pdata = scv.pAnnData.from_file("example_report.txt", source="diann")
pdata.describe()
```

For more detailed examples, check out the [Tutorials](tutorials/index.md).
## Documentation Layout
- **[Tutorials](tutorials/index.md)** – Step-by-step guides for importing, filtering, plotting, and running enrichment.  
- **[API Reference](reference/index.md)** – Full function documentation for `pAnnData` and helper utilities.  
- **[Developer Notes](dev/index.md)** – Guidelines for contributing, testing, and extending scviz.  

## Example Workflow
1. **Import data**  
   Load DIA-NN or PD reports into a `pAnnData` object.

2. **Filter and preprocess**  
   Apply peptide/protein filters, normalize intensities, and impute missing values.

3. **Analyze**  
   Run differential expression, generate visualizations, and perform enrichment analyses.

4. **Export**  
   Save processed data and results for downstream applications.

---

🧪 *scviz is developed for research in single-cell and spatial proteomics, supporting reproducible and scalable analysis.*  

:books: To learn more, explore the [Tutorials](tutorials/index.md) or dive into the [API Reference](reference/index.md).


# OLD STUFF
## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

Testing emojis :beers: and soccer.

Lorem ipsum[^1] dolor sit amet, consectetur adipiscing elit.[^2]

[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
[^2]:
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.
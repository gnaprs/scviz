# Welcome to scpviz
![logo](assets/300ppi/logo_black_label@300x.png#only-light){ align=right width=150 }
![logo](assets/300ppi/logo_white_label@300x.png#only-dark){ align=right width=150 }

**scpviz** is a Python package for single-cell and spatial proteomics data analysis, built around a custom `pAnnData` object.  
It extends the [AnnData](https://anndata.readthedocs.io/) ecosystem with proteomics-specific functionality, enabling seamless integration of proteins, peptides, and relational data.

## Features
<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Set up in 5 minutes__

    ---

    Install [`scpviz`](#) with [`pip`](#) and get up
    and running in minutes

    [:octicons-arrow-right-24: Getting started](tutorials/installation.md)

-   :fontawesome-brands-markdown:{ .lg .middle } __Quickstart__

    ---

    Check out the quickstart guide for a run through import, basic preprocessing and quick visualization

    [:octicons-arrow-right-24: Quickstart](tutorials/quickstart.md)

-   :material-format-font:{ .lg .middle } __In-depth Tutorials__

    ---

    Step-by-step guides for importing, filtering, plotting, and running enrichment.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-scale-balance:{ .lg .middle } __API Reference__

    ---

    Full function documentation for `pAnnData` and helper utilities.  

    [:octicons-arrow-right-24: API Reference](reference/index.md)

</div>


- **Single-cell proteomics support**: Store protein and peptide quantifications in AnnData-compatible structures.  
- **Relational mapping**: Track protein–peptide connectivity using a dedicated RS matrix.  
- **Analysis tools**: Filtering, normalization, imputation, and differential expression (DE) tailored for proteomics.  
- **Functional enrichment**: Integrated [STRING](https://string-db.org/) queries for GSEA and PPI networks.  
- **Custom plotting**: Publication-ready plots (abundance, PCA/UMAP, clustermaps, rank-quant plots, etc.).  
- **API utilities**: Retrieve annotations from UniProt, cache mappings, and manage large datasets efficiently.  

## Getting Started
Check out the [quickstart](tutorials/quickstart.md) guide.
For more detailed examples, check out the [Tutorials](tutorials/index.md).

- **[Developer Notes](dev/contributing.md)** – Guidelines for contributing, testing, and extending scpviz.  


---

*scpviz is developed for research in single-cell and spatial proteomics, supporting reproducible and scalable analysis.*  

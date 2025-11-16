---
title: 'scpviz: A Python bioinformatics toolkit for Single-cell Proteomics and multi-omics analysis'
tags:
  - Python
  - single-cell
  - proteomics
  - bioinformatics
authors:
  - name: Marion Pang
    orcid: 0000-0002-0158-2976
    corresponding: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Baiyi Quan
    orcid: 0000-0001-6313-4274
    affiliation: 2
  - name: Ting-Yu Wang
    orcid: 0000-0002-9014-6825
    affiliation: 2
  - name: Tsui-Fen Chou
    orcid: 0000-0003-2410-2186
    corresponding: true
    affiliation: "1, 2"
affiliations:
 - name: Division of Biology and Biological Engineering, California Institute of Technology, 1200 E. California Blvd, Pasadena, CA 91125
   index: 1
 - name: Proteome Exploration Laboratory, Beckman Institute, California Institute of Technology, 1200 E. California Blvd, Pasadena, CA 91125
   index: 2
date: 15 November 2025
bibliography: paper.bib

---

# Summary
Proteomics seeks to characterize protein dynamics by measuring both protein abundance and post-translational modifications (PTMs), such as phosphorylation, acetylation, and ubiquitination, which regulate protein activity, localization, and interactions. In bottom-up proteomics workflows, proteins are enzymatically digested into peptides that are measured as spectra, from which these peptide-spectrum matches (PSMs) are aggregated to infer protein-level identifications and quantitative abundance estimates. Analyzing the two levels of data at both the peptide level (short fragments observed directly) and the protein level (assembled from peptide evidence) in tandem is crucial for translating raw measurements into biologically interpretable results.

Single-cell proteomics extends these approaches to resolve protein expression at the level of individual cells or microdissected tissue regions. Such data are typically sparse, with many missing values, and are generated within complex experimental designs involving multiple classes of samples (e.g., cell type, treatment, condition). These properties distinguish single-cell proteomics from bulk experiments and create unique challenges in data processing, normalization, and interpretation. The single-cell transcriptomics community has established a mature ecosystem for managing similar challenges, exemplified by the `scanpy` package [@SCANPY:2018] and the broader `scverse` ecosystem [@virshupScverseProjectProvides2023a]. Building on these foundations, `scpviz` extends the AnnData data structure to the domain of proteomics, supporting a complete analysis pipeline from raw peptide-level data to protein-level summaries and downstream interpretation through differential expression, enrichment analysis, and network analysis. The core of `scpviz` is the `pAnnData` class, an `AnnData`-affiliated data structure specialized for proteomics. Together, these components make `scpviz` a comprehensive and extensible framework for single-cell proteomics. By combining flexible data structures, reproducible workflows, and seamless integration with the `AnnData`, `scanpy` and extended `scverse` ecosystem, the package enables researchers to efficiently connect peptide-level evidence to protein-level interpretation, thereby accelerating methodological development and biological discovery in proteomics.

# Statement of need
Although general-purpose data analysis frameworks such as `scanpy` [@SCANPY:2018] and the broader `scverse` ecosystem have become indispensable for single-cell transcriptomics, comparable tools for proteomics remain limited. Existing proteomics software often focus on specialized tasks (e.g., peptide identification or spectrum assignment) and do not provide a unified framework for downstream analysis of peptide- and protein-level data within single-cell and spatial contexts. 

`scpviz` addresses these gaps by offering an integrated system for the complete proteomics workflow, from raw peptide-level evidence to protein-level summaries and biological interpretation. It is designed for computational biologists and proteomics researchers working with low-input or single-cell datasets from data sources such as Proteome Discoverer or DIA-NN[@demichevDIANNNeuralNetworks2020a].

At the core of scpviz is the `pAnnData` class, an `AnnData`-affiliated data structure specialized for proteomics. It organizes peptide (`.pep`) and protein (`.prot`) `AnnData` objects alongside supporting attributes such as `.summary`, `.metadata`, `.rs` matrices (protein–peptide relationships), and `.stats`. This design allows users to move flexibly between between peptides and proteins while maintaining compatibility with established Python libraries for data science and visualization.

Beyond data organization, `scpviz` implements proteomics-specific operations, including filtering (e.g., requiring proteins supported by at least two unique peptides), normalization and imputation methods tailored for sparse datasets, and visualization tools such as PCA (Principal Component Analysis), UMAP (Uniform Manifold Approximation and Projection for Dimension Reduction), clustermaps, and abundance plots. For downstream interpretation, it integrates with UniProt for annotation and string-db for enrichment and network analysis [@szklarczykSTRINGDatabase20232023; @snelSTRINGWebserverRetrieve2000; @mcinnesUMAPUniformManifold2018a]. The framework also incorporates single-cell proteomics–specific normalization strategies such as directlfq [@ammarAccurateLabelFreeQuantification2023], ensuring robust quantification across heterogeneous samples. Finally, pAnnData objects interface seamlessly with scanpy [@SCANPY:2018] and other ecosystem tools such as harmony [@korsunskyFastSensitiveAccurate2019a], enabling direct incorporation into established single-cell workflows.

The design philosophy of `scpviz` emphasizes both usability and extensibility. General users can rely on its streamlined API to import, process, and visualize single-cell proteomics data without deep programming expertise, while advanced users can extend the framework to accommodate custom analysis pipelines. The package has already been applied in published papers and preprints [@Pang:2025; @DuttaPang:2025; @Uslan:2025; @duttaParkinsonsDiseaseModeling2025] as well as manuscripts in preparation, and it has been incorporated into graduate-level training to illustrate how proteomics workflows parallel to those in single-cell transcriptomics.

The applications of `scpviz` span diverse areas of life sciences research, from studying protein dynamics and signaling pathways to integrating proteomics with transcriptomics for multi-omics analysis. By bridging the gap between raw mass spectrometry data and systems-level interpretation, `scpviz` provides a versatile and reproducible platform for advancing single-cell and spatial proteomics.

# Acknowledgements

We thank Pierre Walker for his many insightful discussions and guidance. We also acknowledge support from the A*STAR BS-PhD Scholarship. The Proteome Exploration Laboratory is partially supported by the Caltech Beckman Institute Endowment Funds.

<!-- If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit. --> 

<!-- Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# References
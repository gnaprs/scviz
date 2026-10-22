# TODO

**Priority key:** H = high | M = medium | L = backlog  
**Status key:** [ ] = todo | [~] = in progress | [x] = done

H4C1: add to histone housekeeping list

---

## High priority (H)

Quick view of what to work on. Full tables below.
*Regenerate the H table from sections below when priorities change.*

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| H | [x] | base-1 | Allow users to input "obs" for their own metadata, own transformer for pdata.neighbor |
| H | [ ] | base-4 | Make wildcard '*' for plotting volcano plots — pair w search annotation? |
| H | [~] | viz-3 | Add gsea — one for PCA, one for ssGSEA |
| H | [ ] | ref-4 | Docstring hover: hovering over functions doesn't show docstring all the time (pdata class inherits mixin; .plotting/.utils are fine) — fix ASAP |
| H | [ ] | bug-6 | Parquet import: verify peptide-to-protein splicing (ProtA; ProtB). Maybe apply directlfq fix to import? |
| H | [x] | doc-1 | Tutorial: integrate with scanpy features |
| H | [~] | mod-5 | Web app for other users — check out dash-webapp branch by Baiyi |

---

## Features

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| H | [x] | base-1 | Allow users to input "obs" for their own metadata, own transformer for pdata.neighbor |
| L | [ ] | base-2 | Add sharedPeptides function on get_CV() |
| L | [ ] | base-3 | Concating multiple datasets? — see ad.concat, maybe use adata.obs_names_make_unique() |
| H | [ ] | base-4 | Make wildcard '*' for plotting volcano plots — pair w search annotation? |
| L | [ ] | viz-1 | Dot plot: expression of indicated genes in three clusters (dot size = % cells per cluster; color = cluster average normalized expression). Partially doable w scanpy. |
| M | [ ] | viz-2 | Compare two comparisons: log(fc) vs log(fc) with significance coloring |
| H | [x] | viz-3 | Add gsea — one for PCA, one for ssGSEA |
| M | [x] | viz-4 | Hypothesis testing — ANOVA with Tukey and BH correction |
| M | [x] | viz-5 | plot_grouped_heatmap + plot_clustered_heatmap (publication heatmaps) |
| M | [ ] | enr-1 | String DB values rank API — check API key; per-user or local copy in package? [API help](https://string-db.org/cgi/help?subpage=api%23valuesranks-enrichment-api) |
| L | [ ] | feat-qc-1 | Consider QC metrics beyond what we already have |

---

## Enhancements

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| M | [ ] | enh-1 | When updating .summary/obs, move prot/pep details to the right (prioritize metadata). Affects users, fix soon. |
| L | [x] | enh-2 | get_pca_importance: accept pdata input (and prot/pep) — get uns['pca'] and var_names from it. Pair with PCA GSEA? |
| L | [ ] | enh-3 | Double-check peptide export format: Gene Name, Peptide name, peptide AA start/end in protein, Charge, sample-name columns for intensities |
| L | [ ] | enh-4 | Check for DIA/DDA and suggest preprocessing methods? |
| M | [ ] | enh-5 | Fuzzy match for get_abundance matches |
| M | [ ] | enh-6 | Move Found In and Significant In from var/obs? Too many columns — or express more concisely (e.g. .obsm/.varm) |
| L | [ ] | enh-7 | Add modification print for diann import |
| L | [ ] | enh-8 | When impute errors on wrong obs column, pretty-format the error message |

---

## Refactor & code quality

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| M | [~] | ref-1 | Sync DE for adata and pdata into a combined workflow, and volcano code |
| L | [ ] | ref-2 | Standardize internal terminology: `classes` vs `class_types` for sample-level grouping |
| L | [ ] | ref-3 | Add typing to variables — low priority, should be easy |
| H | [ ] | ref-4 | Docstring hover: hovering over functions doesn't show docstring all the time (pdata class inherits mixin; .plotting/.utils are fine) — fix ASAP |

---

## Bugs & known fixes

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| L | [ ] | bug-1 | Clustermap bug when linkage |
| M | [ ] | bug-2 | plot_abundnace_housekeeping throws error when no housekeeping gene is found |
| L | [ ] | bug-3 | Verify we sync rs and filter rs matrix for every filter operation (not only filter sample by condition?) |
| H | [ ] | bug-4 | Parquet import: verify peptide-to-protein splicing (ProtA; ProtB). Maybe apply directlfq fix to import? |



---

## Documentation

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| H | [x] | doc-1 | Tutorial: integrate with scanpy features |
| L | [ ] | doc-2 | QC tutorial |
| L | [ ] | doc-3 | search_annotation tutorial |
| M | [ ] | doc-4 | Finish under-construction tutorials: `imputation.md`, `de.md`, `enrichment.md` |



---

## Maintenance

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
|  | [ ] | maint-1 | Todo for maintenance goes here |

---

## New modules

| Priority | Status | ID | Description |
|----------|--------|----|-------------|
| L | [~] | mod-1 | Peptide sequence characteristics (hydrophobicity, etc.) [peptide_param module]. Can work with Baiyi. |
| L | [ ] | mod-2 | Correlation visualization [protein corr module] |
| M | [ ] | mod-3 | Omics module for comparing adata? — build on existing transcriptomics x proteomics functions |
| L | [ ] | mod-4 | Alphamap peptide mapping visualization |
| H | [~] | mod-5 | Web app for other users — check out dash-webapp branch by Baiyi |

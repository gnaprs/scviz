*This tutorial is still under construction*

# single cell

Handling single cell data is slightly different from bulk data. More sparse, missing values, so need to handle normalization and imputation more carefully. Here, we go through some typical workflows for single cell proteomics (field is still being established!)

## preprocessing
### filter proteins
Filter by significance, valid genes and for unique profiles (due to single cell data having high missing values, a lot of peptides are often missing - this means often isoforms will have duplicated abundance profiles since there are no unique peptides distinguishing the isoforms.)

```py
pdata_filtered = pdata.filter_prot(valid_genes=True, unique_profiles=True)
pdata_filtered = pdata_filtered.filter_prot_significance()
```

I also suggest doing a 40% filter for interested groups followed by minimum value imputation
```py
pdata_filtered = pdata.filter_prot_found(groups='sample', min_ratio = 0.4)
pdata_filtered.impute(method='min', min_value = 0.2)
```

### normalization
typically bulk proteomics is median normalization

there's also directlfq algorithm that we can use, is implemented by doing
```
pdata.normalize(method='directlfq')
```

!!! note
    this algorithm will create files in the workspace, and also might take awhile

## visualize data
typically done with umap

can also use tsne plot

## using scanpy
need to first use cleanup function to make data clean for scanpy (expects 0, not NaNs, will throw error otherwise)
```
pdata.clean_X()
```

scanpy expects AnnData objects, so we send in the .prot objects (after we have done filtering with eg pep)

```
pdata.filter_rs(unique_pep=2)
prot = pdata.prot
sc.tsne(prot)
```
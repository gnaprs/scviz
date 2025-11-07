*This tutorial is still under construction*
# single cell

Handling single cell data is slightly different from bulk data. More sparse, missing values, so need to handle normalization and imputation more carefully. Here, we go through some typical workflows for single cell proteomics (field is still being established!)

## preprocessing
### filter proteins
we suggest doing a 40% filter followed by minimum value imputation

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
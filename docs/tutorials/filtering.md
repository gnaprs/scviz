# Tutorial 2: Filtering and Normalization

Learn how to filter proteins/peptides.

All filter functions return a **copy** of the `pAnnData` object unless `inplace=True` is specified.  
See the [Filtering](filtering.md) tutorial for more options.

You can also filter individual proteins using `pdata.filter_prot()` and `pdata.filter_rs()`, which support both condition-based and RS-based filtering.

``` py title="Filter/Subset"
# Basic filtering (keep minimum prot count of 4000)  # (1)
pdata = pdata.filter_sample(min_prot=4000)

# Filter for high confidence proteins (q < 0.01)
pdata = pdata.filter_prot_significant()
```

1.  You can also filter individual proteins using `pdata.filter_prot()` and `pdata.filter_rs()`, which supports both condition-based and RS-based filtering.

## filter_prot
Filters protein
maybe show table

### by signifiance (FDR q value)

### by presence in samples

### by unique Peptide

```python
# Keep proteins with at least 2 unique peptides
pdata.filter_prot(condition="unique_peptides >= 2")
```

---
## filter_sample

### by condition

### by value

### by queries

```python
# Advanced query mode on sample metadata
pdata.filter_sample(values="treatment == 'drug' and timepoint == '24h'", query_mode=True)
```

Add example to advanced query for filter

(A and B) or C

---


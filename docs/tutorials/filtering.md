*This tutorial is still under construction*
# Tutorial 2: Filtering and Normalization

Learn how to filter proteins and peptides in your dataset.

All filter functions return a **copy** of the `pAnnData` object unless `inplace=True` is specified.

There are three main filtering functions:

* `pdata.filter_prot()` – filters proteins from the dataset  
  * `pdata.filter_prot_found()` – sub-function for proteins found within specified samples  
  * `pdata.filter_prot_significant()` – sub-function for proteins significant within specified samples  
* `pdata.filter_sample()` – filters samples from the dataset  
* `pdata.filter_rs()` – RS-based filtering (e.g., filtering by unique peptides)

---

## `filter_prot()`

Filter protein data based on metadata conditions or accession lists (protein or gene names).

### Condition-based filtering

``` py title="Filter by metadata condition"
condition = "Protein FDR Confidence: Combined == 'High'"
pdata.filter_prot(condition=condition)
```

``` py title="Substring match on protein description"
condition = "Description includes 'p97'"
pdata.filter_prot(condition=condition)
```

``` py title="Numerical condition on metadata"
condition = "Score > 0.75"
pdata.filter_prot(condition=condition)
```

``` py title="Keep proteins with ≥2 unique peptides"
pdata.filter_prot(condition="unique_peptides >= 2")
```

> **Note:** For `condition`, the first variable must match a column name in `prot.var`. Otherwise, an error will be raised.

---

### Accession-based filtering (accession list or gene names)

``` py title="Filter by specific accessions or genes"
accessions = ['GAPDH', 'P53']
pdata.filter_prot(accessions=accessions)
```

### Valid genes

This removes rows with missing gene names and resolves duplicate gene names by appending numeric suffixes.

``` py title="Filter proteins with valid genes only"
pdata.filter_prot(valid_genes=True)
```

---

### Unique profiles

Removes rows with duplicate abundance profiles across samples (typically for isoforms with no distinguishing peptides).

``` py title="Filter proteins with unique abundance profiles"
pdata.filter_prot(unique_profiles=True)
```

---

!!! tip
    Multiple filters can be combined in a single call. For example:

    ``` py
    condition = "Score > 0.75"
    pdata.filter_prot(condition=condition, valid_genes=True)
    ```

## `filter_prot_found()`

Filter proteins or peptides based on *"Found In"* detection across samples or groups.

This method filters features by checking whether they are found in a minimum number or proportion of samples, either at the group level (e.g., biological condition) or based on individual files.

``` py title="Filter proteins found in both 'groupA' and 'groupB' groups with AND logic"
pdata.filter_prot_found(group=["groupA", "groupB"], min_count=2, match_any=False)
```

| Group A |           | Group B |           | Result |
|:-------:|:----------|:-------:|:----------|:-------:|
| F1 | F2 | F3 | F4 | |
| 🟩 | 🟩 | 🟩 | ⚪ | ✅ Kept (found ≥2 per group) |
| 🟩 | ⚪ | ⚪ | ⚪ | ❌ Filtered (not enough in group B) |

---

``` py title="Filter proteins found in all three input files"
pdata.filter_prot_found(group=["F1", "F2", "F3"])
```

| F1 | F2 | F3 | Result |
|:--:|:--:|:--:|:------:|
| 🟩 | 🟩 | 🟩 | ✅ Kept |
| 🟩 | ⚪ | 🟩 | ❌ Filtered |

---

``` py title="Filter proteins found in files of a specific sub-group"
pdata.annotate_found(classes=['group', 'treatment'])
pdata.filter_prot_found(group=["groupA_control", "groupB_treated"])
```

``` py title="Filter by class column, based on a minimum ratio (e.g., at least 50% in each cell line)"
pdata.filter_prot_found(group="cellline", min_ratio=0.5, match_any=False)
```
> **Note**: A true `match_any` flag retain proteins significant in *any* group/file (OR logic). If `False`, requires *all* groups/files to be significant (AND logic). The flag defaults to `True`.

---

## `filter_prot_significant()`

Filter proteins based on significance across samples or groups using FDR thresholds.

This method filters proteins by checking whether they are significant (e.g., `PG.Q.Value < 0.01`) in a minimum number or proportion of samples, either per file or grouped.

``` py title="Filter proteins significant in both 'groupA' and 'groupB' groups"
pdata.filter_prot_significant(group=["groupA", "groupB"], min_count=2)
```

``` py title="Filter proteins significant in all three input files"
pdata.filter_prot_significant(group=["F1", "F2", "F3"])
```

``` py title="Filter proteins significant in files of a specific sub-group"
pdata.annotate_significant(classes=['group', 'treatment'])
pdata.filter_prot_significant(group=["groupA_control", "groupB_treated"])
```



---

#### Example: `annotate_found(classes=['group','treatment'])` then `filter_prot_found(group=["groupA_control","groupB_treated"])`

| groupA_control | groupB_treated | Result |
|:---------------:|:--------------:|:-------:|
| 🟩 | 🟩 | ✅ Kept |
| 🟩 | ⚪ | ❌ Filtered |

---

#### Example: `filter_prot_found(group="cellline", min_ratio=0.5)`

| Cellline A |      | Cellline B |      | Ratio | Result |
|:-----------:|:-----|:-----------:|:-----|:------:|:-------:|
| F1 | F2 | F3 | F4 | ≥ 0.5 | ✅ Kept |
| 🟩 | ⚪ | 🟩 | ⚪ | 0.5 | ✅ |
| 🟩 | ⚪ | ⚪ | ⚪ | 0.25 | ❌ |


### by unique Peptide

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


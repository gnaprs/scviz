# Tutorial 3: Imputation and Normalization
## Imputation
Missing values are common in proteomics. scviz provides several imputation methods.

---

### KNN Imputation

```python
pdata.impute(method="knn", n_neighbors=5)
```

---

### Group-wise Imputation

```python
pdata.impute(method="group_mean", groupby="condition")
```

---
#
## Checking Imputation Stats

```python
pdata.stats["imputation"]
```

📊 *This dictionary stores how many values were imputed per group or overall.*

## Normalization
and apply normalization strategies
— adjust for sample effects (global, reference feature, or directLFQ).  
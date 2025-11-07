*This tutorial is still under construction*

# Tutorial 3: Imputation and Normalization
## Imputation
Missing values are common in proteomics. scpviz provides several imputation methods.

!!! note
    Pre-processing functions like normalize() and impute() act directly on the pAnnData object rather than returning a copy.
    By default, the processed data are written to the active .X layer unless a new layer name is specified.

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

## Normalization

```python
# Normalize intensities by global median
pdata.normalize(method="global")

# Normalize to reference proteins
pdata.normalize(method="reference_feature", reference_columns=["ACTB", "GAPDH"])
```

💡 *Note: Normalization choices can affect downstream DE and enrichment.*

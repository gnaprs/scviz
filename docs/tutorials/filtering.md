# Tutorial: Filtering and Normalization

Learn how to filter proteins/peptides and apply normalization strategies.

---

## Filtering by Peptide Support

```python
# Keep proteins with at least 2 unique peptides
pdata.filter_prot(condition="unique_peptides >= 2")
```

---

## Filtering Samples with Queries

```python
# Advanced query mode on sample metadata
pdata.filter_sample(values="treatment == 'drug' and timepoint == '24h'", query_mode=True)
```

---

## Normalization

```python
# Normalize intensities by global median
pdata.normalize(method="global")

# Normalize to reference proteins
pdata.normalize(method="reference_feature", reference_columns=["ACTB", "GAPDH"])
```

💡 *Note: Normalization choices can affect downstream DE and enrichment.*

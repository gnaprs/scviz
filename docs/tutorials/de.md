# Tutorial 5: Differential Expression (DE)

Run DE analysis at the protein or peptide level.
— test across groups (mean, pairwise, peptide-level).  
---

## Protein-Level DE

```python
de_results = pdata.de(group1="treated", group2="control", on="protein")
de_results.head()
```

---

## Fold Change Strategies

```python
# Mean-based fold change
pdata.de(group1="treated", group2="control", method="mean")

# Pairwise protein-level median
pdata.de(group1="treated", group2="control", method="protein_pairwise")

# Peptide-level median (via RS matrix)
pdata.de(group1="treated", group2="control", method="peptide_pairwise")
```

💡 *Different strategies may be useful depending on noise and sample size.*

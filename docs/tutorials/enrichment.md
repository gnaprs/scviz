# Tutorial: Enrichment and Networks

Use STRING integration for enrichment and protein–protein interaction (PPI) analysis.

---

## Functional Enrichment (GSEA / GO)

```python
pdata.enrichment_functional(on="protein", source="de_results")
```

---

## PPI Enrichment

```python
pdata.ppi_enrichment(on="protein", source="de_results")
```

---

## Viewing Results

```python
pdata.list_enrichments()
```

🔗 *Each enrichment run will provide a STRING link for interactive exploration.*

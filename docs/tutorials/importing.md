# Tutorial: Importing Data

This tutorial shows how to import DIA-NN or Proteome Discoverer (PD) outputs into a `pAnnData` object.

---

## Loading DIA-NN Reports

```python
import scviz as scv

# Load DIA-NN report
pdata = scv.pAnnData.from_file("example_diann_report.txt", source="diann")

pdata.describe()
```

---

## Loading Proteome Discoverer (PD) Reports

```python
pdata = scv.pAnnData.from_file("example_pd_report.txt", source="pd")
pdata.describe()
```

---

## Metadata Parsing

Sample metadata (columns in `.obs`) can be inferred directly from filenames:

```python
pdata.obs.head()
```

💡 *Tip: If filenames follow different formats, scviz will suggest possible `.obs` columns or default to generic labels.*

...
---
➡️ Next: [Filtering and Normalization](filtering.md)

# Quickstart

This quickstart tutorial demonstrates a minimal end-to-end workflow using **scviz**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-repo>/blob/main/tutorials/quickstart.ipynb)

[Download notebook](quickstart.ipynb)

---


```python
import scviz

# Import DIA-NN data
pdata = scviz.import_diann("report.tsv")

# Basic filtering (keep minimum prot count of 1200)
pdata = pdata.filter_sample(min_prot=1000)

# Basic filtering (keep proteins with ≥2 unique peptides)
pdata = pdata.filter_rs(min_unique_peptides_per_protein=2)

# Normalize (median scaling)
pdata = pdata.normalize(method="median")

# Impute missing values (KNN-based)
pdata = pdata.impute(method="knn")

# Quick visualization
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,4))
pdata.plot_abundance(ax, namelist=["GAPDH", "UBE4B"], classes="treatment")
plt.show()
```

---

## Next steps
For a complete walkthrough of the analysis workflow — from importing data to enrichment and network analysis — see the [Tutorial Index](index.md).

- For details on **data import options**, see [Importing Data](importing.md).  
- For more on **filtering**, see [Filtering](filtering.md).  
- For advanced visualization, try the [Plotting tutorial](plotting.md).  

!!! tip
    Most plotting functions accept a `matplotlib.axes.Axes` object as the first argument.  
    This makes it easy to embed scviz plots in multi-panel figures.

*This tutorial is still under construction*

# Tutorial 4: Plotting

Generate publication-ready plots with scpviz.
— generate abundance plots, PCA/UMAP, clustermaps, raincloud, volcano plots.  


Most plotting functions accept a `matplotlib.axes.Axes` object as the first argument, allowing seamless integration into multi-panel figures.

---

## Abundance Plots

```python
pdata.plot_abundance(
    namelist=["ACTB", "GAPDH"],
    classes="condition",
    order=["control", "treated"]
)
```
The `plot_abundance()` function automatically selects between barplots and violin plots (with inner points) depending on the number of samples per group.
---

## PCA and UMAP

```python
pdata.plot_pca(classes="celltype")
pdata.plot_umap(classes="condition")
```

---

## Clustermap

```python
pdata.plot_clustermap(namelist=["TP53", "VIM", "MAPT"], classes="condition")
```

🎨 *Colors automatically follow sample classes, but you can customize palettes.*

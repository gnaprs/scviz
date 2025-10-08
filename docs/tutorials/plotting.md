# Tutorial 4: Plotting

Generate publication-ready plots with scviz.
— generate abundance plots, PCA/UMAP, clustermaps, raincloud, volcano plots.  
---

## Abundance Plots

```python
pdata.plot_abundance(
    namelist=["ACTB", "GAPDH"],
    classes="condition",
    order=["control", "treated"]
)
```

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

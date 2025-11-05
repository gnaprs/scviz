# Quickstart

This quickstart tutorial demonstrates a minimal end-to-end workflow using **scpviz**.

---
## Import
Start by importing
```py
from scpviz import pAnnData as pAnnData
```

`scpviz` currently supports two data formats, more incoming. but for now, there's Proteome Discoverer (Thermo Fisher format) and reports from DIA-NN. 

=== "Proteome Discoverer"

    ``` py title="Import Proteome Discoverer data"
    obs_columns = ['Sample','cellline','ko','condition','duration']
    pdata = pAnnData.import_data(
        source_type='pd', 
        prot_file = '../assets/pd_prot.txt', 
        obs_columns=obs_columns)
    ```

    <div class="result" markdown>

    ``` py title="output"
    🧭 [USER] Importing data of type [pd]
    --------------------------
    Starting import [Proteome Discoverer]

    Source file: ../assets/pd_prot_short.txt / None
    Number of files: 12
    Proteins: 6

            🔄 [UPDATE] Updating summary [recompute]: Recomputed metrics and refreshed `.summary` from `.obs`.
        ℹ️ [INFO] Using global q-values for 'prot' significance annotation.

        ✅ [OK] pAnnData object is valid.
        ✅ [OK] Import complete. Use `print(pdata)` to view the object.
    --------------------------
    ```

    </div>

=== "DIA-NN"

    ``` py title="Import DIA-NN data"
    obs_columns = ['user', 'date', 'ms', 'acquisition', 'faims', 'column', 'gradient', 'amount', 'region', 'rep']
    pdata = pAnnData.import_data(
        source_type='diann', 
        report_file = '../assets/diann_report.parquet', 
        obs_columns=obs_columns)
    ```

    <div class="result" markdown>

    ``` py title="output"
    🧭 [USER] Importing data of type [diann]
    --------------------------
    Starting import [DIA-NN]

    Source file: ../assets/diann_report.parquet
    Number of files: 12
    Proteins: 2251
    Peptides: 7688


    ℹ️ RS matrix: (2251, 7688) (proteins × peptides), sparsity: 99.95%
    - Proteins with ≥2 *unique* linked peptides: 1217/2251
    - Peptides linked to ≥2 proteins: 158/7688
    - Mean peptides per protein: 3.53
    - Mean proteins per peptide: 1.03
        ✅ [OK] pAnnData object is valid.
        ✅ [OK] Import complete. Use `print(pdata)` to view the object.
    --------------------------
    ```

    </div>


## Pre-processing

All filter functions return a **copy** of the `pAnnData` object by default, unless `inplace=True` is specified.
``` py title="Filter/Subset"
# Basic filtering (keep minimum prot count of 4000)  # (1)
pdata = pdata.filter_sample(min_prot=4000)

# Filter for high confidence proteins (q < 0.01)
pdata = pdata.filter_prot_significant()
```

1.  You can also filter individual proteins using `pdata.filter_prot()` and `pdata.filter_rs()`, which supports both condition-based and RS-based filtering.

Other pre-processing functions act on `pAnnData` object, and by default `.X` is set to the new processed layer.

``` py title="Normalization and imputation"
# Normalize (median scaling)
pdata.normalize(method="median")

# Impute missing values (e.g. minimum value)
pdata.impute(method="min")
```

<div class="result" markdown>

``` py title="output"
🧭 [USER] Global normalization using 'median'. Layer will be saved as 'X_norm_median'.
     ✅ Normalized all 71 samples.
     ℹ️ Set protein data to layer X_norm_median.
🧭 [USER] Global imputation using 'min'. Layer saved as 'X_impute_min'. Minimum scaled by 1.
     ✅ 104032 values imputed.
     ℹ️ 71 samples fully imputed, 0 samples partially imputed, 0 skipped feature(s) with all missing values.
     ℹ️ Set protein data to layer X_impute_min.
```

</div>

## Quick visualization
We can do a quick visualization of particular proteins of interest in our samples.
```py
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4,4))
pdata.plot_abundance(ax, namelist=["GAPDH", "VCP"], classes=["ko","cellline"])
plt.show()
```

<div class="result" markdown>
<figure markdown="span">
  ![Protein abundance plot](../assets/quickstart_1.png)
  <figcaption></figcaption>
</figure>

</div>

Alternatively, we can have a look at the PCA embeddings. Other embeddings like UMAP and T-SNE can also be plotted.
```py
fig, ax = plt.subplots(figsize = (4,4))
ax = scplt.plot_pca(ax, pdata, classes=["ko","cellline"])
```

<div class="result" markdown>
<figure markdown="span">
  ![Protein abundance plot](../assets/quickstart_2.png)
  <figcaption></figcaption>
</figure>

</div>

## Differential expression


!!! tip
    Most plotting functions accept a `matplotlib.axes.Axes` object as the first argument. This makes it easy to embed scpviz plots in multi-panel figures.
---

## Next steps
For a complete walkthrough of the analysis workflow — from importing data to enrichment and network analysis — see the [Tutorial Index](index.md).

- For details on **data import options**, see [Importing Data](importing.md).  
- For more on **filtering**, see [Filtering](filtering.md).  
- For advanced visualization, try the [Plotting tutorial](plotting.md).  


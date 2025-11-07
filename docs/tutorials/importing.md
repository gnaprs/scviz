*This tutorial is still under construction*

# Tutorial 1: Importing Data

This tutorial shows how to import DIA-NN or Proteome Discoverer (PD) outputs into a `pAnnData` object.
 — load DIA-NN or PD outputs into `pAnnData`.  
---

## Loading DIA-NN Reports

```python
import scpviz as scv

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


note that for PD, there is only global FDR data unlike for DIA-NN

### before version 3.2 i.e. PD 2.5
can also import, using the same format
---

## Metadata Parsing
— extract `.obs` columns from filenames or reports.  
Sample metadata (columns in `.obs`) can be inferred directly from filenames:

```python
pdata.obs.head()
```

- any updates to .summary will be automatically pushed to `.prot.obs` and `.pep.obs` (if available). User will be prompted when necessary to run `pdata.update_summary()`, typically if I (author of package) can't tell if its intentional/

*If filenames follow different formats, scpviz will suggest possible `.obs` columns or default to generic labels.*

In this case, I recommend making a parsing function - e.g. 
`parse_filenames` and reassigning to `.summary`. 

```py
def parse_filename_index(df):
    """
    Parses the index of a DataFrame assumed to be filenames into structured metadata columns.

    Expected filename format (delimited by "_"):
        [0] date
        [1] gradient
        [2] sample_id
        [3] size
        [4] confirmation
        [5] thickness
        [6] sample
        [7] organism
        [8] region
        [9] well_position

    Args:
        df (pd.DataFrame): DataFrame with index containing delimited filenames.

    Returns:
        pd.DataFrame: Original DataFrame with added metadata columns.
    """
    colnames = [
        'date',
        'gradient',
        'sample_id',
        'size',
        'confirmation',
        'thickness',
        'sample',
        'organism',
        'region',
        'well_position'
    ]

    parts = df.index.to_series().str.split('_', expand=True)
    if parts.shape[1] != len(colnames):
        raise ValueError(f"Expected {len(colnames)} parts after splitting index, got {parts.shape[1]}")

    df_parsed = df.copy()
    for i, col in enumerate(colnames):
        df_parsed[col] = parts.iloc[:, i]
    return df_parsed

```

---

## Export results
— save processed datasets, DE tables, or plots.

...
---
➡️ Next: [Filtering and Normalization](filtering.md)

# Tutorial 2: Filtering

[![Download Notebook](https://img.shields.io/badge/download-filtering__tutorial-blue?logo=icloud&style=flat-square)](https://github.com/gnaprs/scpviz/raw/main/docs/tutorials/filtering.ipynb)
[![Open In Colab](https://img.shields.io/badge/open%20in-colab-yellow?logo=googlecolab&style=flat-square)](https://colab.research.google.com/github/gnaprs/scpviz/blob/main/docs/tutorials/filtering.ipynb)

Learn how to filter proteins and peptides in your dataset.

All filter functions return a **copy** of the `pAnnData` object unless `return_copy=False` is specified.

There are three main filtering functions:

* `pdata.filter_prot()` – filters proteins from the dataset  
    * `pdata.filter_prot_found()` – sub-function for proteins found within specified samples  
    * `pdata.filter_prot_significant()` – sub-function for proteins significant within specified samples  
* `pdata.filter_sample()` – filters samples from the dataset  
* `pdata.filter_rs()` – RS-based filtering (e.g., filtering by unique peptides)

---

## `filter_prot()`

Filter protein data based on metadata conditions or accession lists (protein or gene names).

!!! tip
    Multiple filters can be combined in a single call. For example:

    ``` py
    condition = "protein_quant > 0.75"
    pdata_filtered = pdata.filter_prot(condition=condition, valid_genes=True, unique_profiles=True)
    ```

### Condition-based filtering
A condition string to filter protein metadata. Supports:
- Standard comparisons, e.g. `"Protein FDR Confidence: Combined == 'High'"`
- Substring queries using `includes`, e.g. `"Description includes 'p97'"`

``` py title="Filter by matching condition metadata"
condition = "Protein FDR Confidence: Combined == 'High'"
pdata_filtered = pdata.filter_prot(condition=condition)
```

``` py title="Filter by numerical condition on metadata"
pdata_filtered = pdata.filter_prot(condition="unique_peptides >= 2")
```

> **Note:** For `condition`, the first variable must match a column name in `prot.var`. Otherwise, an error will be raised.


``` py title="Substring match on protein description"
condition = "Description includes 'VCP'"
pdata_filtered = pdata.filter_prot(condition=condition)
```

<div class="result" markdown>

```
pdata_filtered.prot.var
```

| Accession | | Description|  | Genes |
|:-----------|:-------------|:-------------|:-------------|:------|
| **P55072** |...| Transitional endoplasmic reticulum ATPase OS=Homo sapiens OX=9606 GN=**VCP** PE=1 SV=4 |...| VCP |
| **Q96JH7** |...| Deubiquitinating protein **VCP**IP1 OS=Homo sapiens OX=9606 GN=**VCP**IP1 PE=1 SV=2 |...| VCPIP1 |
| **Q8NHG7** |...| Small **VCP**/p97-interacting protein OS=Homo sapiens OX=9606 GN=SVIP PE=1 SV=1 |...| SVIP |
| **Q9H867** |...| Protein N-lysine methyltransferase METTL21D OS=Homo sapiens OX=9606 GN=**VCP**KMT PE=1 SV=2 |...| VCPKMT |

</div>

---

### Accession-based filtering (accession list or gene names)
Accession-based filtering accepts both UniProt accessions as well as gene names. `pAnnData` objects automatically search the UniProt API for primary gene names upon import, and stores these in the object.

``` py title="Filter by specific accessions or genes"
accessions = ['GAPDH', 'P53']
pdata_filtered = pdata.filter_prot(accessions=accessions)
```

By default, a **gene name** resolves to **one** protein accession (the last isoform in `.var` order if several share the same `Genes` value). Set `all_matches=True` to retain **every** isoform that matches the gene symbol:

``` py title="Filter by gene name, keeping all isoforms"
pdata_filtered = pdata.filter_prot(accessions=['GAPDH'], all_matches=True)
```

When multiple accessions match, an `[INFO]` message lists the resolved isoforms. The same `all_matches` flag is available on [`get_abundance()`](../reference/pAnnData/editing_mixin.md#src.scpviz.pAnnData.editing.EditingMixin.get_abundance) for extracting abundance from every isoform.

### Valid genes

This removes rows with missing gene names and resolves duplicate gene names by appending numeric suffixes.

``` py title="Filter proteins with valid genes only"
pdata_filtered = pdata.filter_prot(valid_genes=True)
```

<div class="result" markdown>

| Accession | Genes (before) | Genes (after) |
|:-----------|:----------------|:----------------|
| **P12345** | GAPDH | GAPDH |
| **P23456** | ACTB | ACTB |
| **P34567** | NaN | ❌ (removed) |
| **P45678** | HSP90AA1 | HSP90AA1 |
| **P45679** | HSP90AA1 | HSP90AA1_2 |
| **P56789** | TUBB | TUBB |
| … | … | … |

</div>

---

### Unique profiles

Removes rows with duplicate abundance profiles across samples (typically for isoforms with no distinguishing peptides). 

!!! tip
    Recommended to use for single-cell data, which has higher data sparsity and missing values in peptides, which frequently leads to duplicated profiles.

``` py title="Filter proteins with unique abundance profiles"
pdata_filtered = pdata.filter_prot(unique_profiles=True)
```

*For more information, see the API documentation for [filter_prot()](https://gnaprs.github.io/scpviz/reference/pAnnData/filtering_mixin/#src.scpviz.pAnnData.filtering.FilterMixin.filter_prot)*

---

## `filter_prot_found()`

Filter proteins or peptides based on *"Found In"* detection across samples or groups.

This method filters features by checking whether they are found in a minimum number or proportion of samples, either at the group level (e.g., biological condition) or based on individual files.
> **Note**: A true `match_any` flag retain proteins true in *any* group/file (OR logic). If `False`, requires *all* groups/files to be true (AND logic). The flag defaults to `True`.

### `groups` and `match_any`

``` py title="Filter proteins found in both cell lines (match_any=False, AND logic)"
pdata_filtered = pdata.filter_prot_found(group="cellline", min_count=2, match_any=False)
```

In this example, the class column `"cellline"` contains two groups: **A** and **B**.  
Proteins must be detected in at least two samples **within each cell line** to be retained.

| Cell line A |           | Cell line B |           | Result |
|:------------:|:----------|:------------:|:----------|:-------:|
| F1 | F2 | F3 | F4 | |
| 🟩 | 🟩 | 🟩 | ⚪ | ✅ Kept (found ≥2 per cell line) |
| 🟩 | 🟩 | ⚪ | ⚪ | ❌ Filtered (not enough in cell line B), kept if `match_any=True` |

> The `group` parameter refers to a **sample class column** (e.g., `"cellline"`, `"treatment"`, `"condition"`).  
> Each unique value in that column (e.g., `A`, `B`) is treated as a separate subgroup.

``` py title="Filter proteins found in any cell line (match_any=True, OR logic, ratio ≥ 0.4)"
pdata_filtered = pdata.filter_prot_found(group="cellline", min_ratio=0.4, match_any=True)
```

This example uses the class column `"cellline"` containing **A** and **B**.  
Proteins are retained if they are found in **at least 40% of samples within *any one* cell line**.

| Cell line A |      |      | Cell line B |      | Ratio (A, B) | Result |
|:------------:|:-----|:-----|:------------:|:-----|:--------------:|:-------:|
| F1 | F2 | F3 | F4 | F5 | | |
| 🟩 | 🟩 | ⚪ | ⚪ | ⚪ | (0.67, 0.00) | ✅ Kept (≥0.4 in A) |
| 🟩 | ⚪ | ⚪ | 🟩 | ⚪ | (0.33, 0.50) | ✅ Kept (≥0.4 in B) |
| ⚪ | ⚪ | 🟩 | ⚪ | ⚪ | (0.33, 0.00) | ❌ Filtered (<0.4 in both) |

> With `match_any=True`, **OR logic** is applied across groups —  
> a protein passes if it meets the minimum ratio threshold in *any one* subgroup.

### Filter by found in file-list

``` py title="Filter proteins found in all three input files"
pdata_filtered = pdata.filter_prot_found(group=["F1", "F2", "F3"])
```

| F1 | F2 | F3 | Result |
|:--:|:--:|:--:|:------:|
| 🟩 | 🟩 | 🟩 | ✅ Kept (found in all 3 files)|
| 🟩 | ⚪ | 🟩 | ❌ Filtered (not found in File 2) |

---

``` py title="Filter proteins found in files of a specific sub-group with AND logic"
pdata.annotate_found(classes=['group', 'condition'])
pdata_filtered = pdata.filter_prot_found(group=['groupA_control', 'groupB_treated'],min_ratio=0.5,match_any=False)
```

| groupA_control | groupA_treated | groupB_control | groupB_treated | Result |
|:---------------:|:--------------:|:---------------:|:--------------:|:-------:|
| 🟩 | ⚪ | ⚪ | 🟩 |✅ Kept (Found in both specified groups) |
| 🟩 | 🟩 | 🟩 | ⚪ | ❌ Filtered (Not found in `groupB_treated` group) |

``` py title="Filter by class column, based on a minimum ratio (e.g., at least 50% in each cell line)"
pdata_filtered = pdata.filter_prot_found(group="cellline", min_ratio=0.5, match_any=False)
```

| Cell line A |      | Cell line B |   |   | Ratio | Result |
|:-----------:|:-----|:-----------:|:-----|:-----|:------:|:-------:|
| F1 | F2 | F3 | F4 | F5 | ≥ 0.5 |  |
| 🟩 | ⚪ | 🟩 | 🟩 | ⚪ |  (0.5, 0.66) | ✅ |
| 🟩 | ⚪ | 🟩 | ⚪ | ⚪ |  (0.5, 0.33) | ❌ (✅ if `match_any=True`) |
| 🟩 | ⚪ | ⚪ | ⚪ | ⚪ | (0.25, 0) | ❌ |

*For more information, see the API documentation for [filter_prot_found()](https://gnaprs.github.io/scpviz/reference/pAnnData/filtering_mixin/#src.scpviz.pAnnData.filtering.FilterMixin.filter_prot_found)*

---

## `filter_prot_significant()`

Filter proteins based on significance across samples or groups using FDR thresholds. 

This method filters proteins by checking whether they are significant (e.g., `PG.Q.Value < 0.01`) in a minimum number or proportion of samples, either per file or grouped.

The grouping logic is akin to that of `filter_prot_found()`.

!!! warn
    Only DIA-NN files contain per-sample specific q-values. For PD files, use `pdata.filter_prot_significant()` to filter based on global q-values.

> **Note**: A true `match_any` flag retain proteins significant in *any* group/file (OR logic). If `False`, requires *all* groups/files to be significant (AND logic). The flag defaults to `True`.


``` py title="Filter proteins significant in all 'cellline' groups ('celline_A' and 'celline_B')"
pdata_filtered = pdata.filter_prot_significant(group=["cellline"], min_count=2)
```

``` py title="Filter proteins significant in all three input files"
pdata_filtered = pdata.filter_prot_significant(group=["F1", "F2", "F3"])
```

``` py title="Filter proteins significant in files of a specific sub-group"
pdata.annotate_significant(classes=['group', 'treatment'])
pdata_filtered = pdata.filter_prot_significant(group=["groupA_control", "groupB_treated"])
```

*For more information, see the API documentation for [filter_prot_significant()](https://gnaprs.github.io/scpviz/reference/pAnnData/filtering_mixin/#src.scpviz.pAnnData.filtering.FilterMixin.filter_prot_significant)*

## `filter_sample()`

Filter samples in a `pAnnData` object based on categorical, numeric, identifier-based, or POI abundance criteria.  
Accepts **exactly one** of the following arguments:

- `values`: A dictionary or list of dictionaries specifying class-based filters (e.g., treatment, cellline).  
- `condition`: A logical condition string evaluated against summary-level numeric metadata (e.g., protein count).  
- `min_prot`: Minimum number of proteins detected per sample (shortcut for `condition="protein_count >= N"`).  
- `file_list`: A list of sample or file names to retain.  
- `exclude_file_list`: A list of sample or file names to remove.  
- `poi`: A protein or peptide of interest; retain samples whose POI abundance meets a threshold.

---

### Filter by value

Categorical metadata filtering allows selection of samples based on `.obs` or `.summary` fields such as treatment, cell line, or condition.  
This supports:

- A **single dictionary**, e.g. `{'cellline': 'A'}`  
- A **list of dictionaries** for multiple matching cases, e.g. `[{...}, {...}]`  
- **Exact matching**: 
    * `exact_cases=True` for strict combination matching across all key–value pairs
    * `exact_cases=False` applies an OR logic within fields and AND logic across fields.

#### `exact_cases=False`

``` py title="Filter by metadata values (exact_cases=False)"
pdata_filtered = pdata.filter_sample(values={'condition': ['kd','sc'], 'cellline': 'A'})
```
When `exact_cases=False`, the logic is **(OR within fields, AND across fields)**.  
This means any sample that matches *any* treatment in `['kd','sc']` **and** has `cellline='A'` is kept.

| Sample | Treatment | Cell line | Match logic | Result |
|:------:|:----------:|:----------:|:-------------|:-------:|
| 1 | sc | A | ✅ treatment in [kd, sc] and ✅ cellline=A | ✅ Kept |
| 2 | kd | A | ✅ treatment in [kd, sc] and ✅ cellline=A | ✅ Kept |
| 3 | sc | B | ✅ treatment in [kd, sc] and ❌ cellline=A | ❌ |
| 4 | kd | B | ✅ treatment in [kd, sc] and ❌ cellline=A | ❌ |

#### `exact_cases=True`

``` py title="Filter with multiple exact matching cases (exact_cases=True)"
pdata_filtered = pdata.filter_sample(
    values=[
        {'condition': 'kd', 'cellline': 'A'},
        {'condition': 'sc', 'cellline': 'B'}
    ],
    exact_cases=True
)
```
When `exact_cases=True`, the logic requires an **exact match** to *one of the full dictionaries*.  
Here, only samples matching either `{treatment: 'kd', cellline: 'A'}` **or** `{treatment: 'sc', cellline: 'B'}` are kept.

| Sample | Treatment | Cell line | Match dictionary | Result |
|:------:|:----------:|:----------:|:----------------:|:-------:|
| 1 | sc | A | ❌ No exact match | ❌ |
| 2 | kd | A | ✅ Matches {'kd','A'} | ✅ Kept |
| 3 | sc | B | ✅ Matches {'sc','B'} | ✅ Kept |
| 4 | kd | B | ❌ No exact match | ❌ |

---

### Filter by condition

Use a logical condition string referencing columns in `pdata.summary`.  
This enables numeric and boolean filtering based on sample-level summary statistics.

``` py title="Filter samples with more than 1000 proteins"
pdata_filtered = pdata.filter_sample(condition="protein_count > 1000")
```

#### Using `min_prot`

Convenience shortcut equivalent to `condition="protein_count >= N"`. Keeps samples with **at least** `N` proteins detected.

``` py title="Keep samples with at least 1000 proteins detected"
pdata_filtered = pdata.filter_sample(min_prot=1000)
```

---

### Filter by file list

Filter samples directly by their file or sample identifiers.

``` py title="Keep specific samples by name"
pdata_filtered = pdata.filter_sample(file_list=['Sample_001', 'Sample_007'])
```

``` py title="Exclude specific samples by name"
pdata_filtered = pdata.filter_sample(exclude_file_list=['Sample_001', 'Sample_007'])
```

---

### Filter by POI abundance

Select samples based on the **raw abundance** of a protein or peptide of interest (POI) in that sample. This is typically a **second step** after a metadata filter (e.g. control samples only):

``` py title="Two-step workflow: metadata, then POI threshold"
control = pdata.filter_sample(values={'treatment': 'control'})
high_poi = control.filter_sample(poi='GAPDH', min_abundance=1e4, match_any=True)
```

`poi` accepts a gene name, UniProt accession, or peptide ID (`on='peptide'`). When several isoforms or peptides match, an `[INFO]` note is appended after the main log message. At least one of `min_abundance` or `max_abundance` is required. If the POI does not resolve, the result is a valid object with **zero samples** retained.

#### `match_any`: AND vs OR across multiple matches

When `poi` resolves to more than one protein isoform or peptide, each **detected** match (abundance &gt; 0, not NaN) is evaluated against the threshold. Undetected matches are **ignored** (they do not pass or fail the sample).

`match_any=False` (**AND**, default): the sample is kept only if **every detected** match passes.

`match_any=True` (**OR**): the sample is kept if **any one detected** match passes.

Example: `poi='GeneX'` maps to isoforms **P1** and **P2**, with `min_abundance=1e4`:

| Sample | P1 abundance | P2 abundance | Detected | `match_any=False` | `match_any=True` |
|:------:|:------------:|:------------:|:--------:|:-----------------:|:----------------:|
| S1 | 1e5 | 1e2 | P1, P2 | ❌ P2 below threshold | ✅ P1 passes |
| S2 | 1e5 | NaN | P1 only | ✅ P1 passes (P2 ignored) | ✅ P1 passes |
| S3 | 1e2 | 1e2 | P1, P2 | ❌ Both below threshold | ❌ Neither passes |
| S4 | NaN | NaN | none | ❌ (see `require_detected`) | ❌ (see `require_detected`) |

``` py title="Require every detected isoform to pass (AND)"
strict = pdata.filter_sample(poi='GAPDH', min_abundance=1e4, match_any=False)
```

``` py title="Require any detected isoform to pass (OR)"
lenient = pdata.filter_sample(poi='GAPDH', min_abundance=1e4, match_any=True)
```

#### `require_detected`: samples with no usable POI signal

Controls what happens when a sample has **no detected** POI values (all NaN or zero for the matched features).

`require_detected=True` (default): if nothing is detected for that sample, it **fails** the filter.

`require_detected=False`: undetected samples **pass** the POI filter (they are not removed by abundance thresholds).

| Sample | P1 | P2 | `require_detected=True`, `match_any=True`, `min=1e4` | `require_detected=False`, same settings |
|:------:|:--:|:--:|:----------------------------------------------------:|:---------------------------------------:|
| S1 | 1e5 | NaN | ✅ P1 passes | ✅ |
| S2 | NaN | NaN | ❌ Nothing detected | ✅ Passes (no detection required) |
| S3 | 1e2 | NaN | ❌ Below threshold | ❌ Below threshold |

#### Other parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `min_abundance` | — | Minimum raw abundance (inclusive) |
| `max_abundance` | — | Maximum raw abundance (inclusive); use with `min_abundance` for a range |
| `layer` | `"X"` | Abundance layer on `.prot` or `.pep` |
| `on` | `"protein"` | `"protein"` or `"peptide"` |

---

### Advanced query mode

Enable **advanced filtering** with `query_mode=True` to execute raw pandas-style queries.  
This interprets `values` or `condition` as a raw `.query()` string evaluated directly on `.obs` or `.summary`.

``` py title="Query .obs metadata using values"
pdata_filtered = pdata.filter_sample(values="cellline == 'AS' and condition == 'kd'", query_mode=True)
```

``` py title="Query .summary metadata using condition"
pdata_filtered = pdata.filter_sample(condition="protein_count > 1000 and protein_quant > 0.9", query_mode=True)
```

Complex logical expressions such as `(A and B) or C` are supported.

---

### Additional flags

- `cleanup`: If `True` (default), remove proteins that become all-NaN or all-zero after sample filtering and synchronize RS/peptide matrices.  
  Set to `False` to retain all proteins (useful for downstream DE analyses requiring consistent feature alignment).

*For more information, see the API documentation for [filter_sample()](https://gnaprs.github.io/scpviz/reference/pAnnData/filtering_mixin/#src.scpviz.pAnnData.filtering.FilterMixin.filter_sample)*

---

## `filter_rs()`

Filter the RS matrix and associated `.prot` and `.pep` data based on peptide–protein relationships.  
This method applies rules for retaining proteins with sufficient peptide evidence and/or removing ambiguous peptides.

### Key Parameters

- **`min_peptides_per_protein`** *(int, optional)* – Minimum total number of peptides required per protein.  
- **`min_unique_peptides_per_protein`** *(int, optional)* – Minimum number of *unique* peptides required per protein.  
- **`max_proteins_per_peptide`** *(int, optional)* – Maximum number of proteins a peptide can map to (peptides exceeding this are removed).  
- **`preset`** *(str or dict, optional)* – Predefined filter presets:  
  - `"default"` → unique peptides ≥ 2  
  - `"lenient"` → total peptides ≥ 2  
  - A dictionary specifying thresholds manually.  
  The default preset is `"default"`.

---

### Filter by unique peptides

``` py title="Filter proteins with ≥2 unique peptides"
pdata_filtered = pdata.filter_rs(min_unique_peptides_per_protein=2)
```

| Protein | Peptide 1 | Peptide 2 | Peptide 3 | Unique peptides | Result |
|:--------|:----------:|:----------:|:----------:|:----------------:|:-------:|
| P001 | 🟩 | 🟩 | ⚪ | 2 | ✅ Kept |
| P002 | 🟩 | ⚪ | ⚪ | 1 | ❌ Filtered |
| P003 | 🟩 | 🟩 | 🟩 | 3 | ✅ Kept |

> Proteins with **fewer than two unique peptides** are removed by default.  
> The filtering operation updates both `.prot` and `.pep` tables and synchronizes their mappings in the RS matrix.

### Inspecting RS mappings (without filtering)

To look up which peptides map to a protein (or vice versa) **without** subsetting the dataset, use the utility functions introduced in the [importing tutorial](importing.md#mapping-accessions-and-peptides):

``` py title="List peptides for an accession"
from scpviz import utils as scutils

df = scutils.get_peptides_for_accessions(pdata, ["GAPDH"])
```

``` py title="List accessions for a peptide ID or sequence"
df = scutils.get_accessions_for_peptides(pdata, [pdata.pep.var_names[0]])
```

Both functions read from the RS matrix and return a DataFrame. They accept gene names or accession IDs on the protein side, and peptide IDs or amino-acid strings on the peptide side. See [get_peptides_for_accessions()](../reference/utils.md#src.scpviz.utils.get_peptides_for_accessions) and [get_accessions_for_peptides()](../reference/utils.md#src.scpviz.utils.get_accessions_for_peptides) in the API reference.

---

*For more information, see the API documentation for [filter_rs()](https://gnaprs.github.io/scpviz/reference/pAnnData/filtering_mixin/#src.scpviz.pAnnData.filtering.FilterMixin.filter_rs)*

# Installation

`scpviz` is distributed as a Python package and can be installed with `pip`.

```bash
pip install scpviz
```

This will install all required dependencies, including `scanpy`, `anndata`, `pandas`, and plotting libraries.

Import and create a `pAnnData` object:

```python
import scpviz as scv

pdata = scv.pAnnData.from_file("example_report.txt", source="diann")
pdata
```

## Development installation

If you want the most up-to-date features, you can install from the GitHub repository:

```bash
git clone https://github.com/<your-org-or-username>/scpviz.git
cd scpviz
pip install -e .
```

This installs `scpviz` in “editable mode,” so any changes to the code will be reflected immediately.

## Dependencies

- Python ≥ 3.9  
- Core scientific stack: `numpy`, `scipy`, `pandas`  
- Data structures: `anndata`, `scanpy`  
- Plotting: `matplotlib`, `seaborn`, `upsetplot`  
- Network and enrichment: `requests` (for UniProt/STRING API access)  

!!! info "Extra dependencies"
    Some tutorials (e.g. STRING enrichment or directLFQ normalization) may require internet access or additional packages. These are noted in the corresponding tutorial pages.

# Installation

`scpviz` is distributed as a Python package and can be installed with `pip`.

```bash
pip install scpviz
```

This will install all required dependencies, including `scanpy`, `anndata`, `pandas`, and common plotting libraries.

## Development installation

To install the latest development version directly from GitHub:

```bash
git clone https://github.com/gnaprs/scpviz.git
cd scpviz
pip install -e .
```

This installs `scpviz` in **editable mode**, so any changes to the code will be reflected immediately. 

If you would like to contribute to `scpviz`, please see the [Contributing Guide](../dev/contributing.md)

## Dependencies

- Python ≥ 3.8
- Core scientific stack: `numpy`, `scipy`, `pandas`  
- Data structures: `anndata`, `scanpy`  
- Plotting: `matplotlib`, `seaborn`, `upsetplot`  
- Network and enrichment: `requests` (for UniProt/STRING API access)  

!!! info "Optional dependencies"
    Some external functions (e.g. Leiden clustering or directLFQ normalization) may require internet access or additional packages. These are noted in the corresponding tutorial pages.

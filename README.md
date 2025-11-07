# scpviz
<img src="https://raw.githubusercontent.com/gnaprs/scpviz/refs/heads/main/docs/assets/300ppi/logo_black_label.png"
 align="right" width="256"/>
 [![DOI](https://zenodo.org/badge/762480088.svg)](https://doi.org/10.5281/zenodo.17362532)

**Build & Tests:**  
[![Build Status](https://github.com/gnaprs/scpviz/actions/workflows/python-package.yml/badge.svg)](https://github.com/gnaprs/scpviz/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/gnaprs/scpviz/branch/main/graph/badge.svg)](https://codecov.io/gh/gnaprs/scpviz)

**Documentation:**  
[![Docs CI](https://github.com/gnaprs/scpviz/actions/workflows/ci.yml/badge.svg)](https://github.com/gnaprs/scpviz/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-v0.3.0-brightgreen.svg)](https://gnaprs.github.io/scpviz)


**scpviz** is a Python package for single-cell and spatial proteomics data analysis, built around a custom `pAnnData` object.  
It extends the [AnnData](https://anndata.readthedocs.io/) ecosystem with proteomics-specific functionality, enabling seamless integration of proteins, peptides, and relational data.

* **Documentation**: https://gnaprs.github.io/scpviz/
* **Conda**: https://anaconda.org/gnaprs/scpviz (FIX)
* **Python Package Index (PyPI)**: https://pypi.org/project/scpviz/

## Getting started
### Installation

`scpviz` requires Python 3.8 or later. It is distributed as a Python package and can be installed with `pip`.

    python3 -m pip install scpviz

This will install all required dependencies, including `scanpy`, `anndata`, `pandas`, and common plotting libraries.

Alternatively, scpviz is available on Conda:

    conda install gnaprs::scpviz

For the most up-to-date version of scpviz, clone the repository and
install the package using pip:

    conda create -n scpviz python=3.8 numpy pandas pip
    conda activate scpviz
    pip install git+https://github.com/gnaprs/scpviz.git@development

### Quickstart

Check out the [quickstart](https://github.com/gnaprs/scpviz/tutorials/quickstart/) guide for a run through import, basic preprocessing and quick visualization

### In-depth Tutorials
For more in-depth guides on importing, filtering, plotting, and running enrichment, see the [tutorials](https://github.com/gnaprs/scpviz/tutorials/).

### API Reference

Full function documentation for the `pAnnData` class and utility modules can be found on our [documentation page](https://github.com/gnaprs/scpviz/reference/).

## Contributing

If you'll like to contribute to `scpviz`, please see the [contributing guidelines](https://github.com/gnaprs/scpviz/dev/contributing/). We welcome contributions from the community to help improve, expand, and document the functionality of scpviz.

## License
`scpviz` was created by Marion Pang. It is licensed under the terms of the MIT license.
# scpviz
<img src="https://raw.githubusercontent.com/gnaprs/scpviz/refs/heads/main/docs/assets/logo.png"
 align="right" width="256"/>
 [![DOI](https://zenodo.org/badge/762480088.svg)](https://doi.org/10.5281/zenodo.17362532)

**Build & Tests:**  
[![Build Status](https://github.com/gnaprs/scpviz/actions/workflows/python-package.yml/badge.svg)](https://github.com/gnaprs/scpviz/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/gnaprs/scpviz/branch/main/graph/badge.svg)](https://codecov.io/gh/gnaprs/scpviz)

**Documentation:**  
[![Docs CI](https://github.com/gnaprs/scpviz/actions/workflows/ci.yml/badge.svg)](https://github.com/gnaprs/scpviz/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-v0.3.0-brightgreen.svg)](https://gnaprs.github.io/scpviz)


`scpviz` is a Python library designed for scientific visualization of single cell proteomics data, with plans to support more data types in the future.

* **Documentation**: https://gnaprs.github.io/scpviz/
* **Conda**: https://anaconda.org/gnaprs/scpviz (FIX)
* **Python Package Index (PyPI)**: https://pypi.org/project/scpviz/

## Features

* [`pAnnData`](https://github.com/bbye98/mdcraft/tree/main/src/mdcraft/algorithm):
Object for protein (`.prot`) and peptide (`.pep`) AnnData objects, along with ...
(Sub bullet points for each mixin?)
    * [`analysis`](https://github.com/bbye98/mdcraft/tree/main/src/mdcraft/analysis):
Serial and parallel data analysis tools built on top of the MDAnalysis
framework.
    * [`editing`]
    * [`identifier`]
    * [`metadata`]
* [`plotting`](https://github.com/bbye98/mdcraft/tree/main/src/mdcraft/fit):
Plotting functions Settings and additional functionality for Matplotlib figures.
* [`utilities`](https://github.com/bbye98/mdcraft/tree/main/src/mdcraft/lammps):
Helper functions for etc

## Getting started
### Installation

scpviz requires Python 3.8 or later.

For the most up-to-date version of scpviz, clone the repository and
install the package using pip:

    conda create -n scpviz python=3.11 numpy pandas pip
    conda activate scpviz
    pip install git+https://github.com/gnaprs/scpviz.git@development

Alternatively, scpviz is available on Conda:

    conda install gnaprs::scpviz

and PyPI:

    python3 -m pip install scpviz

### Tutorial

link to tutorial ipynb for users.

### Tests

After installing, to run the scpviz tests locally, use `pytest`:

    pip install pytest
    cd scpviz
    pytest

## License
`scpviz` was created by Marion Pang. It is licensed under the terms of the MIT license.

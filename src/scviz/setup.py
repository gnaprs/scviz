# scviz is a package for visualizing single cell data
# Author: Marion Pang
# Created: 2024-02-07
# version: 0.1

from setuptools import setup, find_packages

VERSION = '0.1'
DESCRIPTION = 'A package for visualizing single cell data'
LONG_DESCRIPTION = 'A package for visualizing single cell data'

setup(
       # the name must match the folder name 'verysimplemodule'
        name="scviz", 
        version=VERSION,
        author="Marion Pang",
        author_email="<sr_pang@hotmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['pandas','numpy','seaborn','upsetplot'],
        keywords=['single-cell', 'proteomics'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)

# TO DO:
# - add visualization of lcms data (?)
# - 3D PCA plot
# - volcano plot for differential expression
# - GO term enrichment and GSEA

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.collections as clt
from upsetplot import plot, generate_counts, from_contents, query, UpSet
import seaborn as sns
sns.set_theme(context='paper', style='ticks')

from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
from decimal import Decimal

import warnings
warnings.filterwarnings('ignore')
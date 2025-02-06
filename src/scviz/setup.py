# functions to add here
# check for installed packages?

# From HRTAtlas (https://housekeeping.unicamp.br/?homePageGlobal), Hounkpe et al. 2020 https://doi.org/10.1093/nar/gkaa609
# Housekeeping genes for human and mouse in __assets__/MostStable_Human.csv and __assets__/MostStable_Mouse.csv
# These are the most stable genes in the human and mouse genome, respectively, and are used for normalization in scViz

import os
import pandas as pd

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute paths to the assets
human_file_path = os.path.join(current_dir, '__assets__', 'MostStable_Human.csv')
mouse_file_path = os.path.join(current_dir, '__assets__', 'MostStable_Mouse.csv')

# Read the CSV files
housekeeping_HUMAN = pd.read_csv(human_file_path)
housekeeping_MOUSE = pd.read_csv(mouse_file_path)

# set function to get date and time
import datetime
def get_datetime():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# print versions of all dependencies of scviz
def print_versions():
    dependencies = ["numpy","pandas","matplotlib","seaborn",
                    "upsetplot","sklearn","scipy","umap",
                    "adjustText","anndata","requests","matplotlib_venn","pyarrow"]

    print("scViz version: 0.1.0")
    print("Date and time: ", get_datetime())
    print("Dependencies:")
    for package in dependencies:
        try:
            module = __import__(package)
            print(package, module.__version__)
        except ImportError:
            print(package, "not installed")

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
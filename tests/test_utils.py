from scviz import utils as scutils
import pandas as pd
import numpy as np
import pytest

def test_get_data():
    data = pd.read_excel('tests/data.xlsx', sheet_name='Proteins')
    files = scutils.protein_summary(data, variables=['region','amt','phenotype','organism','gradtime'])

    assert files is not None
    assert files.duplicated(subset=['region','amt','phenotype','organism','gradtime','replicate']).sum() == 0

def test_append_norm():
    data = pd.read_excel('tests/data.xlsx', sheet_name='Proteins')
    norm_data = scutils.append_norm(data,'tests/data_norm.csv','tests/data_fn.csv')

    assert norm_data is not None
    assert len(norm_data.columns.tolist()) == len(data.columns.tolist())
    assert not any(['raw' in col for col in norm_data.columns.tolist()])

def test_append_norm_fail():
    data = pd.read_excel('tests/data.xlsx', sheet_name='Proteins')
    with pytest.raises(ValueError):
        scutils.append_norm(data,'tests/data_norm.csv','tests/data_fn.csv',norm_type='linear')

# add test for filter_by_genelist
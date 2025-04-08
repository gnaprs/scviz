import pandas as pd
import numpy as np
import pytest

from scviz.utils import format_class_filter

# def test_get_data():
#     data = pd.read_excel('tests/data.xlsx', sheet_name='Proteins')
#     files = scutils.protein_summary(data, variables=['region','amt','phenotype','organism','gradtime'])

#     assert files is not None
#     assert files.duplicated(subset=['region','amt','phenotype','organism','gradtime','replicate']).sum() == 0

# def test_append_norm():
#     data = pd.read_excel('tests/data.xlsx', sheet_name='Proteins')
#     norm_data = scutils.append_norm(data,'tests/data_norm.csv','tests/data_fn.csv')

#     assert norm_data is not None
#     assert len(norm_data.columns.tolist()) == len(data.columns.tolist())
#     assert not any(['raw' in col for col in norm_data.columns.tolist()])

# def test_append_norm_fail():
#     data = pd.read_excel('tests/data.xlsx', sheet_name='Proteins')
#     with pytest.raises(ValueError):
#         scutils.append_norm(data,'tests/data_norm.csv','tests/data_fn.csv',norm_type='linear')

# def test_format_class_filter_single_class():
#     assert format_class_filter("treatment", "kd") == {'treatment': 'kd'}
#     assert format_class_filter("treatment", ["kd", "sc"], exact_cases=True) == [
#         {'treatment': 'kd'},
#         {'treatment': 'sc'}
#     ]

def test_format_class_filter_multi_class_loose():
    result = format_class_filter(["cellline", "treatment"], ["AS", "kd"])
    expected = {'cellline': 'AS', 'treatment': 'kd'}
    assert result == expected

def test_format_class_filter_multi_class_exact_str():
    result = format_class_filter(["cellline", "treatment"], "AS_kd", exact_cases=True)
    expected = [{'cellline': 'AS', 'treatment': 'kd'}]
    assert result == expected

def test_format_class_filter_multi_class_exact_list_strs():
    result = format_class_filter(["cellline", "treatment"], ["AS_kd", "BE_sc"], exact_cases=True)
    expected = [
        {'cellline': 'AS', 'treatment': 'kd'},
        {'cellline': 'BE', 'treatment': 'sc'}
    ]
    assert result == expected

def test_format_class_filter_mismatched_length_raises():
    with pytest.raises(ValueError, match="must match the number of classes"):
        format_class_filter(["cellline", "treatment"], ["AS"], exact_cases=True)

    with pytest.raises(ValueError, match="must align with the number of classes"):
        format_class_filter(["cellline", "treatment"], ["AS"], exact_cases=False)

def test_format_class_filter_invalid_classes_input():
    with pytest.raises(ValueError, match="Invalid input: `classes`"):
        format_class_filter(None, "kd")
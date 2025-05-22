from itertools import accumulate
from scviz import pAnnData
import pandas as pd
import numpy as np

# filter_prot tests
def test_filter_prot_by_condition(pdata):
    pdata_filt = pdata.filter_prot(condition="unique_peptides > 3", return_copy=True)
    assert pdata_filt.prot.shape[1] < pdata.prot.shape[1]
    assert pdata_filt.prot.var["unique_peptides"].min() > 3

def test_filter_prot_by_accession(pdata):
    acc = list(pdata.prot.var_names[0:2])
    pdata_filt = pdata.filter_prot(accessions=acc, return_copy=True)
    assert pdata_filt.prot.shape[1] == 2
    assert pdata_filt.prot.var_names[0] == acc[0]
    assert pdata_filt.prot.var_names[1] == acc[1]

def test_filter_prot_by_gene_name(pdata):
    gene = pdata.prot.var["Genes"].dropna().values[0]
    pdata_filt = pdata.filter_prot(accessions=[gene], return_copy=True)
    assert gene in pdata_filt.prot.var["Genes"].values

def test_filter_prot_syncs_peptides(pdata):
    orig_rs = pdata.rs.copy()
    orig_prots = np.array(pdata.prot.var_names)
    orig_peps = np.array(pdata.pep.var_names)
    
    pdata_filt = pdata.filter_prot(condition="unique_peptides > 3", return_copy=True)  # should remove nearly all proteins
    # check by calculating rs between prot and pep
    assert pdata_filt.rs.shape[0] < pdata.rs.shape[0]  # rs should be reduced
    assert pdata_filt.rs.shape[1] <= pdata.rs.shape[1]  # rs should be reduced

    # Reconstruct mask of peptides still linked to retained proteins
    retained_peps = set(pdata_filt.pep.var_names)
    prot_mask = np.isin(orig_prots, pdata_filt.prot.var_names)
    rs_filtered = orig_rs[prot_mask, :]
    pep_linkage_counts = np.array(rs_filtered.sum(axis=0)).ravel()

    # Peptides with zero linkage should have been removed
    expected_retained_peps = set(orig_peps[pep_linkage_counts > 0])
    assert retained_peps == expected_retained_peps, "Peptides not correctly synced with protein filtering"

    # Additional check: all retained peptides should have at least one link in the new RS
    new_pep_links = np.array(pdata_filt.rs.sum(axis=0)).ravel()
    assert np.all(new_pep_links > 0), "All peptides in RS should be linked to at least one protein"
    
def test_filter_prot_found_group_mode(pdata):
    grouping_column=["cellline", "treatment"]
    pdata.annotate_found(classes=grouping_column, on="protein")
    
    # Pick a valid group label from the new Found In annotations
    group_cols = [c for c in pdata.prot.var.columns if c.startswith("Found In: ") and "ratio" in c]
    assert group_cols, "No Found In: <group> ratio columns found. Did annotate_found() run correctly?"

    group_name = group_cols[0].replace("Found In: ", "").replace(" ratio", "")

    # Run the filter
    pdata_filt = pdata.filter_prot_found(group=group_name, min_ratio=0.5, on="protein", return_copy=True)

    # Assert some filtering happened
    assert pdata_filt.prot.shape[1] < pdata.prot.shape[1]
    assert f"Found In: {group_name}" in pdata_filt.prot.var.columns

def test_filter_prot_found_file_mode(pdata):
    file_cols = [c for c in pdata.prot.var.columns if c.startswith("Found In: ")]
    assert file_cols, "No file-mode Found In columns found"
    file_names = [c.replace("Found In: ", "") for c in file_cols]
    pdata_filt = pdata.filter_prot_found(group=file_names[0], on="protein", return_copy=True)
    assert pdata_filt.prot.shape[1] < pdata.prot.shape[1]

# prot query tests
# def test_format_prot_query_basic():
#     df = pd.DataFrame(columns=["Protein FDR Confidence: Combined", "Description"])
    
#     # Test backtick formatting
#     cond1 = "Protein FDR Confidence: Combined == 'High'"
#     expected1 = "`Protein FDR Confidence: Combined` == 'High'"
#     assert pAnnData.pAnnData._format_filter_query(cond1, df) == expected1

#     # Test includes conversion
#     cond2 = "Description includes 'p97'"
#     expected2 = "Description.str.contains('p97', case=False, na=False)"
#     assert _format_prot_query(cond2, df) == expected2

#     # Test includes with backtick column
#     cond3 = "Protein FDR Confidence: Combined includes 'High'"
#     expected3 = "`Protein FDR Confidence: Combined`.str.contains('High', case=False, na=False)"
#     assert _format_prot_query(cond3, df) == expected3

# def test_format_prot_query_numeric():
#     df = pd.DataFrame(columns=["Score", "Protein Confidence Level"])
    
#     cond1 = "Score > 0.75"
#     assert _format_prot_query(cond1, df) == "Score > 0.75"

#     cond2 = "Protein Confidence Level >= 95"
#     expected2 = "`Protein Confidence Level` >= 95"
#     assert _format_prot_query(cond2, df) == expected2

# # filter query tests
# @pytest.fixture
# def test_instance():
#     """Provides a dummy instance with access to _format_filter_query."""
#     class Dummy:
#         def _format_filter_query(self, condition, dataframe):
#             return pAnnData._format_filter_query(self, condition, dataframe)
#     return Dummy()

# def test_unquoted_string_auto_quotes(test_instance):
#     df = pd.DataFrame({'amount': ['sc', 'kd']})
#     assert test_instance._format_filter_query("amount == sc", df) == 'amount == "sc"'
#     assert test_instance._format_filter_query("amount != kd", df) == 'amount != "kd"'

# def test_quoted_string_unchanged(test_instance):
#     df = pd.DataFrame({'amount': ['sc', 'kd']})
#     assert test_instance._format_filter_query("amount == 'sc'", df) == "amount == 'sc'"
#     assert test_instance._format_filter_query('amount == "sc"', df) == 'amount == "sc"'

# def test_numeric_filter_untouched(test_instance):
#     df = pd.DataFrame({'score': [1.1, 2.2]})
#     assert test_instance._format_filter_query("score > 1.5", df) == "score > 1.5"

# def test_includes_syntax(test_instance):
#     df = pd.DataFrame({'Protein Description': ['p97 subunit', 'ATPase']})
#     expected = "Protein Description.str.contains('p97', case=False, na=False)"
#     assert test_instance._format_filter_query("Protein Description includes 'p97'", df) == expected

# def test_column_with_spaces_and_special_chars(test_instance):
#     df = pd.DataFrame({'Confidence: Level': ['High', 'Low']})
#     expected = '`Confidence: Level` == "High"'
#     assert test_instance._format_filter_query("Confidence: Level == High", df) == expected

# def test_column_with_spaces_and_quoted_value(test_instance):
#     df = pd.DataFrame({'Confidence: Level': ['High', 'Low']})
#     expected = "`Confidence: Level` == 'High'"
#     assert test_instance._format_filter_query("Confidence: Level == 'High'", df) == expected

# # ----------------------------------------------------------------------
# # unit tests for found in/annotate_found methods:
# def test_found_in_samples_added(pdata):
#     assert all(col.startswith("Found In:") for col in pdata.prot.var.columns if "Found In:" in col)
#     assert all(col.startswith("Found In:") for col in pdata.pep.var.columns if "Found In:" in col)

# def test_found_in_classes_single(pdata):
#     pdata.annotate_found(classes='cellline', on='protein')
#     assert any("Found In: BE" in col for col in pdata.prot.var.columns)

# def test_found_in_classes_combo(pdata):
#     pdata.annotate_found(classes=['cellline', 'treatment'], on='protein')
#     assert any("Found In: BE_kd" in col for col in pdata.prot.var.columns)

# def test_found_ratio_correct_format(pdata):
#     pdata.annotate_found(classes='treatment', on='protein')
#     col = [c for c in pdata.prot.var.columns if 'ratio' in c][0]
#     example = pdata.prot.var[col].iloc[0]
#     assert "/" in example and example.split("/")[1].isdigit()
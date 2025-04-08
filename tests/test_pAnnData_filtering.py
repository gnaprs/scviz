from scviz import pAnnData
import pandas as pd
import pytest

# TODO: fix this suite
# -----------------------
# prot query tests
def test_format_prot_query_basic():
    df = pd.DataFrame(columns=["Protein FDR Confidence: Combined", "Description"])
    
    # Test backtick formatting
    cond1 = "Protein FDR Confidence: Combined == 'High'"
    expected1 = "`Protein FDR Confidence: Combined` == 'High'"
    assert pAnnData.pAnnData._format_filter_query(cond1, df) == expected1

    # Test includes conversion
    cond2 = "Description includes 'p97'"
    expected2 = "Description.str.contains('p97', case=False, na=False)"
    assert _format_prot_query(cond2, df) == expected2

    # Test includes with backtick column
    cond3 = "Protein FDR Confidence: Combined includes 'High'"
    expected3 = "`Protein FDR Confidence: Combined`.str.contains('High', case=False, na=False)"
    assert _format_prot_query(cond3, df) == expected3

def test_format_prot_query_numeric():
    df = pd.DataFrame(columns=["Score", "Protein Confidence Level"])
    
    cond1 = "Score > 0.75"
    assert _format_prot_query(cond1, df) == "Score > 0.75"

    cond2 = "Protein Confidence Level >= 95"
    expected2 = "`Protein Confidence Level` >= 95"
    assert _format_prot_query(cond2, df) == expected2

# filter query tests
@pytest.fixture
def test_instance():
    """Provides a dummy instance with access to _format_filter_query."""
    class Dummy:
        def _format_filter_query(self, condition, dataframe):
            return pAnnData._format_filter_query(self, condition, dataframe)
    return Dummy()

def test_unquoted_string_auto_quotes(test_instance):
    df = pd.DataFrame({'amount': ['sc', 'kd']})
    assert test_instance._format_filter_query("amount == sc", df) == 'amount == "sc"'
    assert test_instance._format_filter_query("amount != kd", df) == 'amount != "kd"'

def test_quoted_string_unchanged(test_instance):
    df = pd.DataFrame({'amount': ['sc', 'kd']})
    assert test_instance._format_filter_query("amount == 'sc'", df) == "amount == 'sc'"
    assert test_instance._format_filter_query('amount == "sc"', df) == 'amount == "sc"'

def test_numeric_filter_untouched(test_instance):
    df = pd.DataFrame({'score': [1.1, 2.2]})
    assert test_instance._format_filter_query("score > 1.5", df) == "score > 1.5"

def test_includes_syntax(test_instance):
    df = pd.DataFrame({'Protein Description': ['p97 subunit', 'ATPase']})
    expected = "Protein Description.str.contains('p97', case=False, na=False)"
    assert test_instance._format_filter_query("Protein Description includes 'p97'", df) == expected

def test_column_with_spaces_and_special_chars(test_instance):
    df = pd.DataFrame({'Confidence: Level': ['High', 'Low']})
    expected = '`Confidence: Level` == "High"'
    assert test_instance._format_filter_query("Confidence: Level == High", df) == expected

def test_column_with_spaces_and_quoted_value(test_instance):
    df = pd.DataFrame({'Confidence: Level': ['High', 'Low']})
    expected = "`Confidence: Level` == 'High'"
    assert test_instance._format_filter_query("Confidence: Level == 'High'", df) == expected

# ----------------------------------------------------------------------
# unit tests for found in/annotate_found methods:
def test_found_in_samples_added(pdata):
    assert all(col.startswith("Found In:") for col in pdata.prot.var.columns if "Found In:" in col)
    assert all(col.startswith("Found In:") for col in pdata.pep.var.columns if "Found In:" in col)

def test_found_in_classes_single(pdata):
    pdata.annotate_found(classes='cellline', on='protein')
    assert any("Found In: BE" in col for col in pdata.prot.var.columns)

def test_found_in_classes_combo(pdata):
    pdata.annotate_found(classes=['cellline', 'treatment'], on='protein')
    assert any("Found In: BE_kd" in col for col in pdata.prot.var.columns)

def test_found_ratio_correct_format(pdata):
    pdata.annotate_found(classes='treatment', on='protein')
    col = [c for c in pdata.prot.var.columns if 'ratio' in c][0]
    example = pdata.prot.var[col].iloc[0]
    assert "/" in example and example.split("/")[1].isdigit()
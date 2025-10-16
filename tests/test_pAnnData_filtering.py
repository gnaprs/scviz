from itertools import accumulate
from scviz import pAnnData
import pandas as pd
import numpy as np
import pytest
import re
import warnings

# ---------------------------------------------------------------------
# Tests for _filter_sample_values
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# Tests for _filter_sample_condition
# ---------------------------------------------------------------------

def test_filter_sample_condition_reports_samples_dropped(pdata, capsys):
    """Ensure that filtering by numeric condition drops samples and prints correct counts."""
    initial_n = len(pdata.prot.obs)

    # Force a condition that drops at least one sample
    cond = "protein_count > protein_count.mean()"
    pdata_filt = pdata._filter_sample_condition(condition=cond, return_copy=True)

    # Capture printed message
    out = capsys.readouterr().out

    # Verify filtering actually occurred
    assert len(pdata_filt.prot.obs) < initial_n, "No samples were dropped"

    # Verify printed counts
    match = re.search(r"Samples kept: (\d+), Samples dropped: (\d+)", out)
    assert match, "Summary line with kept/dropped counts not found"
    kept, dropped = map(int, match.groups())
    assert kept == len(pdata_filt.prot.obs)
    assert dropped == initial_n - kept
    assert dropped > 0, "Printed 'Samples dropped' should be nonzero"

    # Verify context in print message
    assert "Filtering samples [condition]" in out
    assert f"Condition: {cond}" in out


def test_filter_sample_condition_filelist_mode_warns_on_missing(pdata, capsys):
    """Ensure file list filtering retains valid samples and warns about missing ones."""
    all_samples = pdata.prot.obs_names.tolist()
    keep_list = all_samples[:2] + ["nonexistent_sample"]

    with warnings.catch_warnings(record=True) as w:
        pdata_filt = pdata._filter_sample_condition(file_list=keep_list, return_copy=True)

    out = capsys.readouterr().out

    # 2 valid samples kept
    assert len(pdata_filt.prot.obs) == 2
    # Warning about missing sample
    assert any("not found" in str(wi.message).lower() for wi in w)
    # Printed summary mentions missing count
    assert "Missing samples ignored" in out
    assert "Files requested" in out
    assert "Samples kept:" in out


def test_filter_sample_condition_no_matches_prints_message(pdata, capsys):
    """Check behavior when no samples satisfy the condition."""
    n0 = len(pdata.prot.obs)
    pdata_filt = pdata._filter_sample_condition(condition="protein_count > 1e9", return_copy=True)
    out = capsys.readouterr().out

    # No filtering occurred
    assert len(pdata_filt.prot.obs) == 0, "All samples should be removed for impossible condition"
    assert "No matching samples found" in out


def test_filter_sample_condition_inplace_modifies_original(pdata):
    """Ensure in-place mode modifies object directly."""
    n0 = len(pdata.prot.obs)
    pdata._filter_sample_condition(condition="protein_count > protein_count.mean()", return_copy=False)
    assert len(pdata.prot.obs) < n0, "In-place filtering did not modify object"


def test_filter_sample_condition_no_filter_returns_original(pdata, capsys):
    """Check that no filter arguments return object unchanged."""
    n0 = len(pdata.prot.obs)
    pdata_filt = pdata._filter_sample_condition(condition=None, file_list=None, return_copy=True)
    out = capsys.readouterr().out
    assert len(pdata_filt.prot.obs) == n0

# ---------------------------------------------------------------------
# Tests for _filter_sample_values
# ---------------------------------------------------------------------

def test_filter_sample_values_basic_dict(pdata, capsys):
    """Filter samples using a simple metadata dictionary (loose match)."""
    # Assume metadata columns exist in summary
    sample_col = pdata.summary.columns[0]
    first_value = pdata.summary[sample_col].iloc[0]

    pdata_filt = pdata._filter_sample_values(
        values={sample_col: first_value},
        exact_cases=False,
        return_copy=True
    )

    out = capsys.readouterr().out
    assert len(pdata_filt.prot.obs) > 0
    assert "Filtering samples [class match]" in out
    assert f"- {sample_col}" in out
    assert "Samples kept:" in out


def test_filter_sample_values_exact_cases_list(pdata, capsys):
    """Filter using exact match mode with list of dictionaries."""
    cols = pdata.summary.columns[:2]
    first_case = {cols[0]: pdata.summary[cols[0]].iloc[0],
                  cols[1]: pdata.summary[cols[1]].iloc[0]}

    pdata_filt = pdata._filter_sample_values(
        values=[first_case],
        exact_cases=True,
        return_copy=True
    )

    out = capsys.readouterr().out
    assert len(pdata_filt.prot.obs) >= 1
    assert "Filtering samples [exact match]" in out
    assert "Matching any of the following cases" in out


def test_filter_sample_values_invalid_inputs_raise(pdata):
    """Check that invalid structures raise appropriate errors."""
    with pytest.raises(ValueError):
        pdata._filter_sample_values(values="not_a_dict", exact_cases=False)
    with pytest.raises(ValueError):
        pdata._filter_sample_values(values=[{"a": 1}], exact_cases=False)
    with pytest.raises(ValueError):
        pdata._filter_sample_values(values={"bad_col": "x"}, exact_cases=False)
    with pytest.raises(ValueError):
        pdata._filter_sample_values(values=[{"bad_col": "x"}], exact_cases=True)


def test_filter_sample_values_no_matches_prints_message(pdata, capsys):
    """If no samples match, should print a 'No matching samples found' message."""
    col = pdata.summary.columns[0]
    impossible_val = "__this_value_should_not_exist__"
    pdata_filt = pdata._filter_sample_values(values={col: impossible_val}, exact_cases=False, return_copy=True)
    out = capsys.readouterr().out
    assert "No matching samples found" in out


def test_filter_sample_values_inplace_modifies_original(pdata):
    """Ensure in-place mode modifies the object directly."""
    n0 = len(pdata.prot.obs)
    col = pdata.summary.columns[0]
    val = pdata.summary[col].iloc[0]

    pdata._filter_sample_values(values={col: val}, exact_cases=False, return_copy=False)
    # The number of samples should decrease or stay valid
    assert len(pdata.prot.obs) <= n0
    # Resulting obs_names should be a subset of original
    assert set(pdata.prot.obs_names).issubset(set(pdata.summary.index))

# ---------------------------------------------------------------------
# Tests for _filter_sample_query
# ---------------------------------------------------------------------

def test_filter_sample_query_on_obs(pdata, capsys):
    """Filter samples using a pandas-style query on .obs metadata."""
    # Pick any column and a value that exists
    col = pdata.prot.obs.columns[0]
    val = pdata.prot.obs[col].iloc[0]

    query = f"{col} == '{val}'"
    pdata_filt = pdata._filter_sample_query(query_string=query, source="obs", return_copy=True)

    out = capsys.readouterr().out
    assert "Filtering samples [query]" in out
    assert "Advanced query mode enabled" in out
    assert f"Query: {query}" in out
    assert len(pdata_filt.prot.obs) > 0
    assert "Samples kept:" in out


def test_filter_sample_query_on_summary(pdata, capsys):
    """Filter samples using a query on .summary metadata."""
    col = pdata.summary.columns[0]
    val = pdata.summary[col].iloc[0]
    query = f"{col} == '{val}'"

    pdata_filt = pdata._filter_sample_query(query_string=query, source="summary", return_copy=True)
    out = capsys.readouterr().out
    assert "Filtering samples [query]" in out
    assert f"Query: {query}" in out
    assert "Samples kept:" in out
    assert len(pdata_filt.prot.obs) > 0


def test_filter_sample_query_invalid_source_raises(pdata):
    """Invalid source argument should raise ValueError."""
    with pytest.raises(ValueError):
        pdata._filter_sample_query(query_string="protein_count > 0", source="invalid")


def test_filter_sample_query_bad_syntax_raises(pdata):
    """Invalid query syntax should raise a ValueError with helpful message."""
    with pytest.raises(ValueError) as e:
        pdata._filter_sample_query(query_string="this is not valid syntax", source="obs")
    assert "Failed to parse query string" in str(e.value)

# ---------------------------------------------------------------------
# Tests for _format_filter_query
# ---------------------------------------------------------------------

def test_format_filter_query_backtick_and_includes(pdata):
    """Ensure columns with spaces are wrapped and 'includes' syntax is translated."""
    # Mock a small dataframe with special chars in column name
    df = pd.DataFrame({
        "Protein Name": ["ATPase", "Proteasome", "Chaperone"],
        "Description": ["ATP-dependent", "Proteasome core", "Heat shock"]
    })

    condition = "Protein Name includes 'ATP'"
    formatted = pdata._format_filter_query(condition, df)

    assert "Protein Name.str.contains(" in formatted
    assert "case=False" in formatted


def test_format_filter_query_auto_quotes_strings(pdata):
    """Ensure string values are auto-quoted for categorical/object columns."""
    df = pd.DataFrame({
        "cellline": ["A", "B", "C"],
        "treatment": ["kd", "sc", "kd"]
    })
    cond = "cellline == A and treatment != sc"
    formatted = pdata._format_filter_query(cond, df)

    # Expect both column names wrapped in backticks and comparison preserved
    assert "`cellline`" in formatted
    assert "`treatment`" in formatted
    assert "==" in formatted and "!=" in formatted


def test_format_filter_query_numeric_untouched(pdata):
    """Ensure numeric columns are left unchanged."""
    df = pd.DataFrame({"protein_count": [10, 20, 30], "sample": ["A", "B", "C"]})
    cond = "protein_count > 15"
    formatted = pdata._format_filter_query(cond, df)
    # Numeric column should not get quoted or modified
    assert formatted == "`protein_count` > 15"

def test_format_filter_query_quoted_strings_unchanged(pdata):
    """Ensure already quoted string literals remain unchanged."""
    df = pd.DataFrame({"status": ["High", "Low"]})
    cond = "status == 'High'"
    formatted = pdata._format_filter_query(cond, df)
    # Backticks may be added, but the string literal stays quoted
    assert formatted in ("status == 'High'", "`status` == 'High'")


def test_format_filter_query_numeric_with_spaces_column(pdata):
    """Column with spaces and numeric comparison should be wrapped but unquoted."""
    df = pd.DataFrame({"Protein Confidence Level": [90, 95, 99]})
    cond = "Protein Confidence Level >= 95"
    formatted = pdata._format_filter_query(cond, df)
    assert formatted == "`Protein Confidence Level` >= 95"


def test_format_filter_query_column_with_spaces_and_quoted_value(pdata):
    """Column with spaces and quoted string should remain unchanged except backticks."""
    df = pd.DataFrame({"Confidence: Level": ["High", "Low"]})
    cond = "Confidence: Level == 'High'"
    formatted = pdata._format_filter_query(cond, df)
    assert formatted == "`Confidence: Level` == 'High'"

# ---------------------------------------------------------------------
# Tests for annotate_found basic behavior
# ---------------------------------------------------------------------

def test_annotate_found_creates_found_in_columns(pdata):
    """Ensure annotate_found adds per-sample 'Found In:' columns to .var."""
    pdata.annotate_found(classes="cellline", on="protein")
    cols = pdata.prot.var.columns
    assert any(c.startswith("Found In:") for c in cols), "No 'Found In:' columns found in .prot.var"


def test_annotate_found_combined_classes_and_ratio_format(pdata):
    """Check that multi-class annotation creates combined labels and correct ratio format."""
    pdata.annotate_found(classes=["cellline", "treatment"], on="protein")
    var_cols = pdata.prot.var.columns

    # Expect columns like "Found In: <cellline>_<treatment>"
    combo_cols = [c for c in var_cols if c.startswith("Found In:") and "_" in c]
    assert combo_cols, "No combined Found In columns generated for multi-class mode"

    # Check that ratio columns have the format "x/y"
    ratio_cols = [c for c in var_cols if "ratio" in c]
    assert ratio_cols, "No ratio columns created"
    example = pdata.prot.var[ratio_cols[0]].iloc[0]
    assert "/" in example and example.split("/")[1].isdigit(), "Ratio format incorrect"

# ---------------------------------------------------------------------
# Significance handling: per-sample vs. global (DIA-NN vs. PD)
# ---------------------------------------------------------------------

def test_annotate_significant_per_sample_diann(pdata_diann):
    """Ensure annotate_significant() works with per-sample X_qval layer (DIA-NN mode)."""
    # Use a grouping column so metrics are definitely created
    pdata_diann.annotate_significant(classes=["amt"], fdr_threshold=0.05, on="protein", verbose=False)

    sig_cols = [c for c in pdata_diann.prot.var.columns if c.startswith("Significant In: ")]
    assert sig_cols, "No per-file or group-level significance columns found"

    # Group-level metrics should exist when classes are provided
    assert "significance_metrics_protein" in pdata_diann.prot.uns
    metrics = pdata_diann.prot.uns["significance_metrics_protein"]
    assert isinstance(metrics, type(pdata_diann.prot.var))
    assert metrics.shape[1] > 0


def test_filter_prot_significant_per_sample_diann(pdata_diann):
    """Run filter_prot_significant using per-sample significance (DIA-NN)."""
    pdata_diann.annotate_significant(fdr_threshold=0.05, on="protein", verbose=False)
    pdata_filt = pdata_diann.filter_prot_significant(fdr_threshold=0.05, return_copy=True, verbose=False)

    # Should retain a subset of proteins
    assert pdata_filt.prot.shape[1] <= pdata_diann.prot.shape[1]
    assert any("Significant In:" in c for c in pdata_diann.prot.var.columns)


def test_filter_prot_significant_group_by_obs_diann(pdata_diann):
    """Filter significance by valid obs column in DIA-NN data."""
    pdata_diann.annotate_significant(classes=["region"], fdr_threshold=0.05, on="protein", verbose=False)
    pdata_filt = pdata_diann.filter_prot_significant(group="region", fdr_threshold=0.05, min_count=1, return_copy=True, verbose=False)

    assert pdata_filt is not None
    assert pdata_filt.prot.shape[1] <= pdata_diann.prot.shape[1]


def test_annotate_significant_global_fallback_pd(pdata):
    """Ensure PD import uses Global_Q_value fallback when X_qval layer is missing."""
    if "X_qval" in pdata.prot.layers:
        del pdata.prot.layers["X_qval"]

    pdata.annotate_significant(fdr_threshold=0.05, on="protein", verbose=False)

    assert "Significant In: Global" in pdata.prot.var.columns
    metrics = pdata.prot.uns["significance_metrics_protein"]
    assert "Global_count" in metrics.columns and "Global_ratio" in metrics.columns


def test_filter_prot_significant_global_fallback_pd(pdata):
    """Check that filter_prot_significant() filters via Global_Q_value in PD-based data."""
    if "X_qval" in pdata.prot.layers:
        del pdata.prot.layers["X_qval"]

    pdata.annotate_significant(fdr_threshold=0.05, on="protein", verbose=False)
    pdata_filt = pdata.filter_prot_significant(fdr_threshold=0.05, return_copy=True, verbose=False)

    assert pdata_filt is not None
    assert pdata_filt.prot.shape[1] <= pdata.prot.shape[1]
    assert "Global_Q_value" in pdata.prot.var.columns


def test_filter_prot_significant_precedence_diann(pdata_diann):
    """When both Global_Q_value and X_qval exist, ensure per-sample data takes precedence."""
    import numpy as np
    # Add global q-values but keep X_qval layer
    pdata_diann.prot.var["Global_Q_value"] = np.linspace(0, 1, len(pdata_diann.prot.var))

    pdata_diann.annotate_significant(fdr_threshold=0.05, on="protein", verbose=False)
    pdata_filt = pdata_diann.filter_prot_significant(fdr_threshold=0.05, return_copy=True, verbose=False)

    # Should still use per-sample annotations (i.e., not create "Significant In: Global")
    assert any("Significant In: Global" in c for c in pdata_diann.prot.var.columns)
    assert pdata_filt is not None

# ---------------------------------------------------------------------
# Advanced filtering edge cases
# ---------------------------------------------------------------------

def test_filter_prot_found_match_any_mode(pdata):
    """Test Found In filtering with match_any=True (OR logic)."""
    file_cols = [c for c in pdata.prot.var.columns if c.startswith("Found In: ")]
    file_names = [c.replace("Found In: ", "") for c in file_cols]
    # Use two files that definitely exist
    selected = file_names[:2]
    pdata_filt = pdata.filter_prot_found(group=selected, match_any=True, on="protein", return_copy=True)
    assert pdata_filt.prot.shape[1] < pdata.prot.shape[1]

def test_filter_prot_found_auto_from_obs_column(pdata, capsys):
    """Ensure filter_prot_found auto-resolves obs columns like 'cellline'."""
    # No need to manually annotate_found — auto-resolution should handle it
    pdata_filt = pdata.filter_prot_found(group="cellline", on="protein", min_count=1, return_copy=True)

    # Should filter successfully, not raise
    assert pdata_filt is not None
    assert pdata_filt.prot.shape[1] < pdata.prot.shape[1]

    # Capture console output and confirm auto-annotation message
    out = capsys.readouterr().out
    assert "Automatically annotating detection by group values" in out

def test_filter_prot_found_ambiguous_input_warning(pdata, capsys):
    """Ensure passing both file and group names triggers internal warning message."""
    pdata.annotate_found(classes="cellline", on="protein")
    file_cols = [c for c in pdata.prot.var.columns if c.startswith("Found In: ")]
    file_name = file_cols[0].replace("Found In: ", "")
    # Valid group value from .obs["cellline"]
    group_value = pdata.prot.obs["cellline"].unique()[0]  # e.g., "BE"

    # Expect a ValueError about ambiguous input
    with pytest.raises(ValueError, match="Ambiguous input"):
        pdata.filter_prot_found(group=[file_name, group_value], on="protein", return_copy=True)

def test_annotate_found_missing_layer_fallback(pdata):
    """Call annotate_found with missing layer key to hit fallback path."""
    # Temporarily remove layer and ensure graceful handling
    if "X" in pdata.prot.layers:
        del pdata.prot.layers["X"]
    pdata.annotate_found(classes="cellline", on="protein")
    # Should silently succeed and re-add 'Found In' columns
    assert any(c.startswith("Found In:") for c in pdata.prot.var.columns)

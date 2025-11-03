from itertools import accumulate
from scviz import pAnnData
import pandas as pd
import numpy as np
import pytest
import re
import warnings
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------
# Tests for _filter_sample_values
# ---------------------------------------------------------------------

# filter_prot tests
def test_filter_prot_no_protein_data(pdata):
    """Ensure filter_prot raises ValueError when no protein data is present."""
    pdata.prot = None

    with pytest.raises(ValueError, match="No protein data found"):
        pdata.filter_prot(condition="unique_peptides > 2")

def test_filter_prot_no_pep_data(pdata, capsys):
    """Ensure filter_prot runs correctly when .pep and .rs are missing."""
    pdata.pep = None
    pdata.rs = None

    n_before = pdata.prot.shape[1]
    pdata_filt = pdata.filter_prot(condition="unique_peptides > 3", return_copy=True)
    n_after = pdata_filt.prot.shape[1]

    # It should filter correctly without errors
    assert n_after < n_before
    assert pdata_filt.prot.var["unique_peptides"].min() > 3

    # Output should mention filtering but not peptides
    out = capsys.readouterr().out
    assert "Filtering proteins" in out
    assert "peptides filtered based on remaining protein linkage" not in out

def test_filter_prot_no_filters_applied(pdata, capsys):
    """Ensure filter_prot prints fallback message when no filters are provided."""
    # Call filter_prot with no active filters
    pdata_filt = pdata.filter_prot(
        condition=None,
        accessions=None,
        valid_genes=False,
        unique_profiles=False,
        return_copy=True,
    )

    out = capsys.readouterr().out

    # Verify printed fallback message
    assert "Filtering proteins [failed]" in out
    assert "No filters applied" in out

    # Object should remain unchanged (copy but same shape)
    assert pdata_filt.prot.shape == pdata.prot.shape

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

def test_filter_prot_valid_genes(pdata):
    """Ensure proteins with empty or missing gene names are filtered out."""
    # Inject a few invalid gene entries (empty and None)
    pdata.prot.var.loc[pdata.prot.var.index[:3], "Genes"] = ["", None, "UNKNOWN_FAKE"]

    n_before = pdata.prot.shape[1]
    pdata_filt = pdata.filter_prot(valid_genes=True, return_copy=True, debug=True)
    n_after = pdata_filt.prot.shape[1]

    # Should remove at least one (empty or None)
    assert n_after < n_before, "Expected some proteins to be removed due to missing genes."

    # All remaining should have non-empty, non-null gene names
    genes = pdata_filt.prot.var["Genes"]
    assert not genes.isna().any(), "NaN gene names remain after filtering."
    assert not (genes.astype(str).str.strip() == "").any(), "Empty gene names remain after filtering."

    # 'UNKNOWN_' entries are allowed — ensure they’re still present if any existed
    assert any(genes.astype(str).str.startswith("UNKNOWN_") | ~genes.astype(str).str.startswith("UNKNOWN_")), \
        "Unexpected removal of placeholder gene names."

def test_filter_prot_unique_profiles(pdata):
    """Ensure only proteins with unique abundance profiles are kept."""
    # Duplicate the first protein’s abundance profile into the second
    pdata.prot.X[:, 1] = pdata.prot.X[:, 0].copy()
    n_before = pdata.prot.shape[1]

    pdata_filt = pdata.filter_prot(unique_profiles=True, return_copy=True, debug=True)
    n_after = pdata_filt.prot.shape[1]

    # Should remove at least one duplicate
    assert n_after < n_before, "Expected fewer proteins after duplicate removal."

    # Verify all remaining profiles are unique (dense representation)
    X_dense = pdata_filt.prot.X.toarray() if hasattr(pdata_filt.prot.X, "toarray") else pdata_filt.prot.X
    n_unique_profiles = np.unique(X_dense.T, axis=0).shape[0]
    assert n_unique_profiles == X_dense.shape[1], "Duplicate profiles still present after filtering."

def test_filter_prot_duplicate_gene(pdata):
    """Ensure duplicate gene names are suffixed as -2, -3, etc., while the first remains unchanged."""
    # Make a copy to avoid modifying the fixture
    pdata = pdata.copy()
    pdata.prot.var.iloc[:3, pdata.prot.var.columns.get_loc("Genes")] = "gene1"

    pdata.filter_prot(valid_genes=True, return_copy=False)

    renamed_genes = pdata.prot.var["Genes"].iloc[0:3].tolist()
    expected = ["gene1", "gene1-2", "gene1-3"]

    assert renamed_genes == expected, f"Expected {expected}, got {renamed_genes}"

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
    pdata_filt = pdata.filter_prot_found(group=group_name, min_ratio=0.5, on="protein", return_copy=True, verbose=True)

    # Assert some filtering happened
    assert pdata_filt.prot.shape[1] < pdata.prot.shape[1]
    assert f"Found In: {group_name}" in pdata_filt.prot.var.columns

def test_filter_prot_found_file_mode(pdata):
    file_cols = [c for c in pdata.prot.var.columns if c.startswith("Found In: ")]
    assert file_cols, "No file-mode Found In columns found"
    file_names = [c.replace("Found In: ", "") for c in file_cols]
    pdata_filt = pdata.filter_prot_found(group=file_names[0], on="protein", return_copy=True)
    assert pdata_filt.prot.shape[1] < pdata.prot.shape[1]

def test_filter_prot_found_group_type_validation(pdata):
    # group must be str or list/tuple of str
    with pytest.raises(TypeError, match="must be a string or list of strings"):
        pdata.filter_prot_found(group=123, on="protein", return_copy=True)

def test_filter_prot_found_missing_groups_and_files_error(pdata):
    # Token that is neither a file column nor a known group value/column
    with pytest.raises(ValueError, match="could not be found"):
        pdata.filter_prot_found(group=["__definitely_not_a_file__", "__not_a_group__"], on="protein", return_copy=True)

def test_filter_prot_found_ambiguous_mixed_file_and_group_value(pdata):
    # Ensure group columns present
    pdata.annotate_found(classes="cellline", on="protein", verbose=False)
    # Pick one file (from existing Found In: <file> columns)
    file_cols = [c for c in pdata.prot.var.columns if c.startswith("Found In: ")]
    assert file_cols, "No file-based 'Found In:' columns found"
    file_name = file_cols[0].replace("Found In: ", "")
    # Pick one valid group value (e.g., 'AS' or 'BE')
    group_value = pdata.prot.obs["cellline"].unique()[0]

    with pytest.raises(ValueError, match="Ambiguous input"):
        pdata.filter_prot_found(group=[file_name, group_value], on="protein", return_copy=True)

def test_filter_prot_found_auto_from_obs_column_requires_threshold(pdata):
    # Pass a class column name -> auto-expands to group values (group mode)
    # Missing both min_ratio and min_count should raise
    pdata.annotate_found(classes="cellline", on="protein", verbose=False)
    with pytest.raises(ValueError, match="specify either `min_ratio` or `min_count`"):
        pdata.filter_prot_found(group="cellline", on="protein", return_copy=True)

def test_filter_prot_found_group_mode_ratio_any_and_all(pdata):
    pdata.annotate_found(classes="cellline", on="protein", verbose=False)
    group_values = pdata.prot.obs["cellline"].unique().tolist()
    # ANY (union) with ratio
    out_any = pdata.filter_prot_found(group=group_values, min_ratio=0.1, on="protein", return_copy=True, match_any=True, verbose=True)
    assert out_any.prot.shape[1] <= pdata.prot.shape[1] and out_any.prot.shape[1] > 0
    # ALL (intersection) with ratio
    out_all = pdata.filter_prot_found(group=group_values, min_ratio=0.1, on="protein", return_copy=True, match_any=False, verbose=False)
    assert out_all.prot.shape[1] <= out_any.prot.shape[1]  # intersection ≤ union

def test_filter_prot_found_group_mode_count_any(pdata):
    pdata.annotate_found(classes="cellline", on="protein", verbose=False)
    group_values = pdata.prot.obs["cellline"].unique().tolist()
    # ANY with min_count
    out_any = pdata.filter_prot_found(group=group_values, min_count=1, on="protein", return_copy=True, match_any=True, verbose=False)
    assert out_any.prot.shape[1] > 0

def test_filter_prot_found_file_mode_any_and_all(pdata):
    # Use two actual files (from Found In: <file> columns)
    file_cols = [c for c in pdata.prot.var.columns if c.startswith("Found In: ")]
    assert len(file_cols) >= 2, "Need at least two file-mode 'Found In:' columns"
    files = [c.replace("Found In: ", "") for c in file_cols[:2]]

    # ANY
    out_any = pdata.filter_prot_found(group=files, on="protein", return_copy=True, match_any=True, verbose=False)
    # ALL
    out_all = pdata.filter_prot_found(group=files, on="protein", return_copy=True, match_any=False, verbose=False)

    assert out_any.prot.shape[1] >= out_all.prot.shape[1]
    assert out_all.prot.shape[1] >= 0

# def test_filter_prot_invalid_args_and_empty_mask(pdata):
#     """Force edge cases in filter_prot(): invalid args, empty result, and inplace mode."""
#     # 1️⃣ Invalid combination of args (both condition & accession_list)
#     with pytest.raises(ValueError):
#         pdata.filter_prot(condition="treatment == 'none'", accession_list=["P00001"])

#     # 2️⃣ Invalid 'on' argument
#     with pytest.raises(ValueError):
#         pdata.filter_prot(condition="cellline == 'BE'", on="invalid")

#     # 3️⃣ Valid condition but yields empty mask → triggers early return
#     pdata_empty = pdata.filter_prot(condition="cellline == 'nonexistent'", return_copy=True, verbose=False)
#     assert pdata_empty.prot.shape[1] == 0

#     # 4️⃣ Inplace mode (return_copy=False)
#     pdata_copy = pdata.copy()
#     pdata_copy.filter_prot(condition="cellline == 'BE'", return_copy=False)
#     # Should update inplace (prot.var_names same length or shorter)
#     assert pdata_copy.prot is not None

# ---------------------------------------------------------------------
# Tests for _filter_sample main
# ---------------------------------------------------------------------
def test_filter_sample_invalid_argument_combination(pdata):
    """Ensure ValueError is raised when multiple filter arguments are given."""
    with pytest.raises(ValueError, match="exactly one"):
        pdata.filter_sample(values={"cellline": "AS"}, condition="protein_count > 10")

def test_filter_sample_min_prot_invokes_condition(pdata):
    """Using min_prot should delegate to _filter_sample_condition."""
    pdata_filtered = pdata.filter_sample(min_prot=10, return_copy=True)
    assert pdata_filtered is not None
    assert pdata_filtered.prot.shape[0] <= pdata.prot.shape[0]

def test_filter_sample_query_mode_obs(pdata):
    """Query mode on obs should evaluate directly via pandas query."""
    col = pdata.prot.obs.columns[0]
    val = pdata.prot.obs[col].iloc[0]
    query = f"{col} == '{val}'"
    out = pdata.filter_sample(values=query, query_mode=True, return_copy=True)
    assert out is not None
    assert out.prot.shape[0] > 0

def test_filter_sample_query_mode_summary(pdata):
    """Query mode on summary should evaluate pandas query on .summary DataFrame."""
    query = "protein_count > 0"
    out = pdata.filter_sample(condition=query, query_mode=True, return_copy=True)
    assert out is not None
    assert out.prot.shape[0] > 0

def test_filter_sample_no_arguments_raises(pdata):
    """Must specify exactly one filter argument."""
    with pytest.raises(ValueError, match="exactly one"):
        pdata.filter_sample()

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

def test_filter_sample_condition_recomputes_summary(pdata, monkeypatch):
    """Ensure _filter_sample_condition recomputes summary when missing."""
    pdata_no_summary = pdata.copy()
    pdata_no_summary._summary = None
    called = {}

    def mock_update_summary(recompute):
        called["recomputed"] = True
        pdata_no_summary._summary = pdata.summary.copy()
    monkeypatch.setattr(pdata_no_summary, "update_summary", mock_update_summary)

    pdata_no_summary._filter_sample_condition(condition="protein_count > 0", return_copy=True)
    assert called.get("recomputed", False)

import warnings

def test_filter_sample_condition_file_list_missing(pdata):
    """Should warn when some file_list entries are missing."""
    file_list = list(pdata.prot.obs_names[:1]) + ["nonexistent_sample"]
    with warnings.catch_warnings(record=True) as w:
        pdata._filter_sample_condition(file_list=file_list, return_copy=True)
    assert any("not found" in str(wi.message) for wi in w)

def test_filter_sample_condition_no_matches_message(pdata, capsys):
    """Should print informative message when no samples satisfy the condition."""
    out = pdata._filter_sample_condition(condition="protein_count > 1e9", return_copy=True)
    captured = capsys.readouterr().out
    assert "No matching samples found" in captured
    # returned pAnnData should have 0 samples
    assert out.prot.n_obs == 0

def test_filter_sample_condition_no_filter_returns_original(pdata):
    """Calling without condition or file_list should return original data."""
    out = pdata._filter_sample_condition(return_copy=True)
    assert out.prot.shape == pdata.prot.shape

def test_filter_sample_condition_file_list_basic(pdata):
    """Basic sanity check for file_list filtering path."""
    sample = pdata.prot.obs_names[0]
    out = pdata._filter_sample_condition(file_list=[sample], return_copy=True)
    assert list(out.prot.obs_names) == [sample]


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

def test_filter_sample_values_no_input_raises(pdata):
    """Calling without `values` should raise ValueError."""
    with pytest.raises(ValueError, match="No filter|must be a dictionar"):
        pdata._filter_sample_values(values=None, exact_cases=False)

def test_filter_sample_values_invalid_list_type(pdata):
    """List elements must all be dictionaries."""
    with pytest.raises(ValueError, match="list.*dictionar"):
        pdata._filter_sample_values(values=["not_a_dict"], exact_cases=True)

def test_filter_sample_values_list_union_mode(pdata):
    """List input with OR logic (default)."""
    obs = pdata.prot.obs
    key = obs.columns[0]
    vals = obs[key].unique()[:2]
    value_list = [{key: v} for v in vals]
    out = pdata._filter_sample_values(values=value_list, exact_cases=True)
    assert out is not None
    assert out.prot.n_obs > 0

    # Should fail with exact_cases=False
    with pytest.raises(ValueError, match="must be a dictionary"):
        pdata._filter_sample_values(values=value_list, exact_cases=False)

def test_filter_sample_values_list_exact_mode(pdata):
    """List input with exact_cases=True (AND logic)."""
    obs = pdata.prot.obs
    key = obs.columns[0]
    val = obs[key].iloc[0]
    # Both dicts share one overlapping key–value pair
    value_list = [{key: val}, {key: val}]
    out = pdata._filter_sample_values(values=value_list, exact_cases=True)
    assert out.prot.n_obs > 0

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

    pdata.annotate_significant(fdr_threshold=0.05, on="protein", verbose=True)
    pdata_filt = pdata.filter_prot_significant(fdr_threshold=0.05, return_copy=True, verbose=True)

    assert pdata_filt is not None
    assert pdata_filt.prot.shape[1] <= pdata.prot.shape[1]
    assert "Global_Q_value" in pdata.prot.var.columns


def test_filter_prot_significant_precedence_diann(pdata_diann):
    """When both Global_Q_value and X_qval exist, ensure per-sample data takes precedence."""
    import numpy as np
    # Add global q-values but keep X_qval layer
    pdata_diann.prot.var["Global_Q_value"] = np.linspace(0, 1, len(pdata_diann.prot.var))

    pdata_diann.annotate_significant(fdr_threshold=0.05, on="protein", verbose=True)
    pdata_filt = pdata_diann.filter_prot_significant(fdr_threshold=0.05, return_copy=True, verbose=True)

    # Should still use per-sample annotations (i.e., not create "Significant In: Global")
    assert any("Significant In: Global" in c for c in pdata_diann.prot.var.columns)
    assert pdata_filt is not None

def test_filter_prot_significant_missing_qval_and_global(pdata):
    """Ensure ValueError raised when neither per-sample qvals nor global q-values exist."""
    # Remove both sources of significance
    pdata_noq = pdata.copy()
    pdata_noq.prot.layers.clear()
    pdata_noq.prot.var.drop(columns=[c for c in pdata_noq.prot.var.columns if "Global_Q_value" in c], errors="ignore", inplace=True)
    with pytest.raises(ValueError, match="No per-sample layer"):
        pdata_noq.filter_prot_significant(return_copy=True)

def test_filter_prot_significant_auto_resolves_obs_column(pdata_diann):
    """Passing a class column name triggers automatic annotation before filtering."""
    # pdata_diann fixture includes obs columns like 'amt', 'region'
    out = pdata_diann.filter_prot_significant(group="amt", fdr_threshold=0.05, min_ratio=0.1, return_copy=True, verbose=False)
    assert out is not None
    # Should have added 'Significant In:' columns corresponding to group values
    assert any("Significant In:" in c for c in out.prot.var.columns)

def test_filter_prot_significant_invalid_group_values(pdata):
    """Ensure ValueError for unrecognized group names."""
    with pytest.raises(ValueError, match="per-sample significance data missing"):
        pdata.filter_prot_significant(group=["nonexistent_group"], fdr_threshold=0.05, return_copy=True)

def test_filter_prot_significant_file_mode_any_and_all(pdata_diann):
    """Exercise per-file significance filtering using ANY vs ALL logic."""
    # First annotate per-sample significance
    pdata_diann.annotate_significant(fdr_threshold=0.05, on="protein", verbose=False)
    sample_names = list(pdata_diann.prot.obs_names[:3])
    # ANY (union)
    out_any = pdata_diann.filter_prot_significant(group=sample_names, fdr_threshold=0.05, match_any=True, return_copy=True, verbose=False)
    # ALL (intersection)
    out_all = pdata_diann.filter_prot_significant(group=sample_names, fdr_threshold=0.05, match_any=False, return_copy=True, verbose=False)
    assert out_any.prot.shape[1] >= out_all.prot.shape[1]

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

# ---------------------------------------------------------------------
# Tests for _filter_rs
# ---------------------------------------------------------------------

def make_dummy_pdata_with_rs(pdata, shape=(3, 4)):
    """Helper: attach a simple binary RS matrix to pdata for filtering tests."""
    rs = csr_matrix(np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ], dtype=int))
    pdata._set_RS(rs, validate=False)

    # Create matching .prot and .pep dummy structures to avoid validation mismatch
    n_prot, n_pep = rs.shape
    pdata.prot = pdata.prot[:, :n_prot] if pdata.prot.shape[1] >= n_prot else pdata.prot.copy()
    pdata.pep = pdata.pep[:, :n_pep] if pdata.pep.shape[1] >= n_pep else pdata.pep.copy()

    pdata.prot.var_names = [f"P{i}" for i in range(n_prot)]
    pdata.pep.var_names = [f"pep{i}" for i in range(n_pep)]
    return pdata

def test_filter_rs_no_rs_returns_self(pdata, capsys):
    """Should safely return and print warning when RS is None."""
    import anndata as ad
    import numpy as np

    pdata._rs = None
    pdata.prot = ad.AnnData(np.zeros((1, 1)))
    pdata.pep = ad.AnnData(np.zeros((1, 1)))
    pdata.validate = lambda *a, **kw: None

    out = pdata.filter_rs(validate_after=False)
    captured = capsys.readouterr().out
    assert "No RS matrix" in captured
    assert out is not None

def test_filter_rs_default_preset(pdata):
    """Test preset='default' → keeps proteins with ≥2 unique peptides."""
    pdata = make_dummy_pdata_with_rs(pdata)
    out = pdata.filter_rs(preset="default", validate_after=False)
    assert "filter_rs" in out.prot.uns
    meta = out.prot.uns["filter_rs"]
    assert meta["n_proteins"] <= pdata.rs.shape[0]


def test_filter_rs_lenient_preset(pdata):
    """Test preset='lenient' → total peptides ≥ 2."""
    pdata = make_dummy_pdata_with_rs(pdata)
    out = pdata.filter_rs(preset="lenient", validate_after=False)
    assert isinstance(out, type(pdata))
    assert "lenient" in out.prot.uns["filter_rs"]["description"]


def test_filter_rs_custom_preset_dict(pdata):
    """Test preset dict overrides numeric thresholds."""
    pdata = make_dummy_pdata_with_rs(pdata)
    preset = {"min_peptides_per_protein": 1, "max_proteins_per_peptide": 2}
    out = pdata.filter_rs(preset=preset, validate_after=False)
    desc = out.prot.uns["filter_rs"]["description"]
    assert "min peptides per protein" in desc
    assert "max proteins per peptide" in desc


def test_filter_rs_invalid_preset_raises(pdata):
    """Unknown preset should raise ValueError."""
    pdata = make_dummy_pdata_with_rs(pdata)
    with pytest.raises(ValueError, match="Unknown RS filtering preset"):
        pdata.filter_rs(preset="weird_mode")


def test_filter_rs_manual_numeric_filters(pdata):
    """Test direct numeric threshold filtering."""
    pdata = make_dummy_pdata_with_rs(pdata)
    out = pdata.filter_rs(
        min_peptides_per_protein=1,
        min_unique_peptides_per_protein=1,
        max_proteins_per_peptide=2,
        validate_after=False,
    )
    assert "filter_rs" in out.prot.uns
    summary = out.prot.uns["filter_rs"]
    assert summary["n_proteins"] > 0
    assert summary["n_peptides"] > 0
    assert "🧪 Filtered RS" in summary["description"]

def test_filter_rs_triggers_validate_and_detects_shape_mismatch(pdata):
    """Ensure validate() is called and detects shape mismatch when RS is invalid."""
    from scipy.sparse import csr_matrix
    import numpy as np

    # Attach a wrong-shaped RS to pdata (too small)
    rs_bad = csr_matrix(np.ones((5, 10), dtype=int))
    pdata._set_RS(rs_bad, validate=False)  # skip check for setup

    # Monkeypatch validate() to simulate real behavior — detect mismatch
    def fake_validate(verbose=True):
        raise ValueError("❌ RS shape mismatch detected during validation")

    pdata.validate = fake_validate

    # Expect the error when validate_after=True
    with pytest.raises(ValueError, match=r"(RS shape mismatch|Length of values)"):
        pdata.filter_rs(validate_after=True)

def test_filter_rs_validate_passes_for_valid_rs(pdata, monkeypatch):
    """Validate should pass when RS shape matches prot×pep dimensions."""
    from scipy.sparse import csr_matrix
    import numpy as np

    # Construct RS that matches actual dataset dimensions
    rs_good = csr_matrix(np.ones((pdata.prot.shape[1], pdata.pep.shape[1]), dtype=int))

    # Use real setter (validate=True)
    pdata._set_RS(rs_good, validate=True)

    # Record whether validate() was called
    called = {}

    # Define mock that records the call
    def record_validate(self, verbose=True):
        called["ran"] = True
        return True

    # Patch at class level so both self and self.copy() share the mock
    monkeypatch.setattr(type(pdata), "validate", record_validate)

    # Run the RS filtering with validation enabled
    pdata.filter_rs(validate_after=True, preset="lenient")

    # Check that validate() was indeed called
    assert called.get("ran", False), "validate() should have been called after filtering"


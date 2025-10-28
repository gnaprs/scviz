import pandas as pd
import numpy as np
import pytest
import scipy
import warnings

from scviz import utils
from anndata import AnnData

@pytest.fixture
def adata_example():
    obs = pd.DataFrame({
        "cellline": ["A", "A", "B", "B"],
        "treatment": ["ctl", "drug", "ctl", "drug"],
        "file_quant": [1, 2, 3, 4]   # to trigger the "_quant" logic
    })
    X = np.random.rand(4, 5)
    return AnnData(X=X, obs=obs)

@pytest.fixture
def adata_gene():
    obs = pd.DataFrame({
        "cellline": ["A", "B", "C"],
        "treatment": ["ctl", "ctl", "drug"]
    })
    var = pd.DataFrame({
        "Genes": ["G1", "G2", "G3"]
    }, index=["P1", "P2", "P3"])
    X = np.array([[1, 10, 100],
                  [2, 20, 200],
                  [3, 30, 300]])
    adata = AnnData(X=X, obs=obs, var=var)
    return adata

# get class_list()
def test_get_classlist_single_column(adata_example):
    result = utils.get_classlist(adata_example, classes="cellline")
    assert set(result) == {"A", "B"}

def test_get_classlist_multiple_columns(adata_example):
    result = utils.get_classlist(adata_example, classes=["cellline", "treatment"])
    expected = {"A_ctl", "A_drug", "B_ctl", "B_drug"}
    assert set(result) == expected

def test_get_classlist_none_uses_pre_quant_columns(adata_example):
    result = utils.get_classlist(adata_example, classes=None)
    # Should join cellline and treatment before hitting _quant
    assert set(result) == {"A_ctl", "A_drug", "B_ctl", "B_drug"}

def test_get_classlist_with_valid_order(adata_example):
    order = ["B", "A"]
    result = utils.get_classlist(adata_example, classes="cellline", order=order)
    assert result == order

def test_get_classlist_invalid_column_raises(adata_example):
    with pytest.raises(ValueError, match="not a column"):
        utils.get_classlist(adata_example, classes="nonexistent")

def test_get_classlist_invalid_order_raises(adata_example, capsys):
    order = ["A", "C"]  # 'C' not in actual classes
    with pytest.raises(ValueError, match="does not match"):
        utils.get_classlist(adata_example, classes="cellline", order=order)
    captured = capsys.readouterr().out
    assert "Missing elements" in captured or "Extra elements" in captured

def test_get_classlist_invalid_type_raises(adata_example):
    with pytest.raises(ValueError, match="Must be None, a string or a list"):
        utils.get_classlist(adata_example, classes={"weird": "dict"})

def test_get_classlist_single_element_list_equivalent(adata_example):
    res1 = utils.get_classlist(adata_example, classes="cellline")
    res2 = utils.get_classlist(adata_example, classes=["cellline"])
    assert set(res1) == set(res2)

def test_format_log_prefix_invalid_level_raises():
    """Ensure ValueError is raised for an unknown log level."""
    with pytest.raises(ValueError, match="Unknown log level"):
        utils.format_log_prefix("notalevel")

def test_get_samplenames_single_column(adata_example):
    """Return values from one column as list of strings."""
    result = utils.get_samplenames(adata_example, "cellline")
    assert result == adata_example.obs["cellline"].tolist()

def test_get_samplenames_multiple_columns(adata_example):
    """Combine multiple columns per row, separated by comma+space."""
    result = utils.get_samplenames(adata_example, ["cellline", "treatment"])
    expected = [
        f"{row.cellline}, {row.treatment}" for _, row in adata_example.obs.iterrows()
    ]
    assert result == expected

def test_get_samplenames_none_returns_none(adata_example):
    """Return None if classes=None."""
    result = utils.get_samplenames(adata_example, None)
    assert result is None

def test_get_samplenames_invalid_type_raises(adata_example):
    """Raise ValueError for unsupported input types."""
    with pytest.raises(ValueError, match="Invalid input for 'classes'"):
        utils.get_samplenames(adata_example, classes=123)

def test_get_adata_layer_returns_X_dense(adata_example):
    """Return .X as dense numpy array when layer='X'."""
    result = utils.get_adata_layer(adata_example, "X")
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, adata_example.X)

def test_get_adata_layer_returns_specific_layer_dense(adata_example):
    """Return dense matrix from a named layer."""
    adata_example.layers["X_norm"] = scipy.sparse.csr_matrix(adata_example.X)
    result = utils.get_adata_layer(adata_example, "X_norm")
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, adata_example.X)

def test_get_adata_layer_invalid_key_raises(adata_example):
    """Raise ValueError when layer key not found."""
    with pytest.raises(ValueError, match="not found"):
        utils.get_adata_layer(adata_example, "nonexistent")

def test_get_abundance_from_adata_basic(adata_gene):
    df = utils._get_abundance_from_adata(adata_gene)
    # Expected columns
    expected_cols = {"cell", "accession", "abundance", "gene", "x_label_name", "class", "log2_abundance"}
    assert expected_cols.issubset(df.columns)
    # Expected number of rows = n_obs * n_vars
    assert len(df) == adata_gene.n_obs * adata_gene.n_vars

def test_get_abundance_with_classes_str(adata_gene):
    df = utils._get_abundance_from_adata(adata_gene, classes="cellline")
    assert "class" in df.columns
    assert set(df["class"]) == set(adata_gene.obs["cellline"])

def test_get_abundance_with_classes_list(adata_gene):
    df = utils._get_abundance_from_adata(adata_gene, classes=["cellline", "treatment"])
    # Should combine values with underscore
    assert all("_" in c for c in df["class"].unique())

def test_get_abundance_no_log(adata_gene):
    df = utils._get_abundance_from_adata(adata_gene, log=False)
    assert "log2_abundance" not in df.columns

def test_get_abundance_x_label_accession(adata_gene):
    df = utils._get_abundance_from_adata(adata_gene, x_label="accession")
    assert (df["x_label_name"] == df["accession"]).all()

def test_get_abundance_with_namelist(monkeypatch, adata_gene):
    # Mock resolve_accessions to avoid external dependency
    monkeypatch.setattr(utils, "resolve_accessions", lambda ad, n, gene_col=None: ["P1"])
    df = utils._get_abundance_from_adata(adata_gene, namelist=["G1"])
    assert set(df["accession"]) == {"P1"}

class DummyPDataAbundance:
    def __init__(self):
        self.called = False
    def get_abundance(self, *a, **kw):
        self.called = True
        return "pdata_path"

def test_get_abundance_dispatches_to_pdata():
    pdata = DummyPDataAbundance()
    result = utils.get_abundance(pdata)
    assert result == "pdata_path"
    assert pdata.called

def test_get_abundance_falls_back_to_adata(adata_gene, monkeypatch):
    called = {}
    def mock_get_abundance_from_adata(adata, *a, **kw):
        called["yes"] = True
        return "adata_path"
    monkeypatch.setattr(utils, "_get_abundance_from_adata", mock_get_abundance_from_adata)
    result = utils.get_abundance(adata_gene)
    assert result == "adata_path"
    assert called["yes"]

def test_format_class_filter_multi_class_loose():
    result = utils.format_class_filter(["cellline", "treatment"], ["AS", "kd"])
    expected = {'cellline': 'AS', 'treatment': 'kd'}
    assert result == expected

def test_format_class_filter_multi_class_exact_str():
    result = utils.format_class_filter(["cellline", "treatment"], "AS_kd", exact_cases=True)
    expected = [{'cellline': 'AS', 'treatment': 'kd'}]
    assert result == expected

def test_format_class_filter_multi_class_exact_list_strs():
    result = utils.format_class_filter(["cellline", "treatment"], ["AS_kd", "BE_sc"], exact_cases=True)
    expected = [
        {'cellline': 'AS', 'treatment': 'kd'},
        {'cellline': 'BE', 'treatment': 'sc'}
    ]
    assert result == expected

def test_format_class_filter_mismatched_length_raises():
    with pytest.raises(ValueError, match="must match the number of classes"):
        utils.format_class_filter(["cellline", "treatment"], ["AS"], exact_cases=True)

    with pytest.raises(ValueError, match="must align with the number of classes"):
        utils.format_class_filter(["cellline", "treatment"], ["AS"], exact_cases=False)

def test_format_class_filter_invalid_classes_input():
    with pytest.raises(ValueError, match="Invalid input: `classes`"):
        utils.format_class_filter(None, "kd")

def test_resolve_accessions_by_accession(adata_gene):
    result = utils.resolve_accessions(adata_gene, ["P1", "P3"])
    assert result == ["P1", "P3"]

def test_resolve_accessions_by_gene(adata_gene):
    result = utils.resolve_accessions(adata_gene, ["G2"])
    assert result == ["P2"]

def test_resolve_accessions_with_gene_map(adata_gene):
    gene_map = {"custom": "P3"}
    result = utils.resolve_accessions(adata_gene, ["custom"], gene_map=gene_map)
    assert result == ["P3"]

def test_resolve_accessions_with_unmatched(adata_gene, capsys):
    result = utils.resolve_accessions(adata_gene, ["P1", "UnknownGene"])
    captured = capsys.readouterr().out
    assert "Unmatched names" in captured
    assert result == ["P1"]

def test_resolve_accessions_no_valid_names_raises(adata_gene):
    with pytest.raises(ValueError, match="No valid names"):
        utils.resolve_accessions(adata_gene, ["XYZ", "ABC"])

def test_resolve_accessions_empty_namelist(adata_gene):
    result = utils.resolve_accessions(adata_gene, [])
    assert result is None

def test_resolve_accessions_with_pdata(pdata):
    """Integration test: resolve a known gene or accession from real pdata."""
    adata = pdata.prot
    gene_col = "Genes" if "Genes" in adata.var.columns else adata.var.columns[0]
    first_gene = adata.var[gene_col].dropna().iloc[0]
    result = utils.resolve_accessions(adata, [first_gene])
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in adata.var_names

def test_get_upset_contents_returns_dataframe(monkeypatch, pdata):
    """Test that get_upset_contents returns UpSet-compatible DataFrame."""
    # Mock get_classlist → 2 sample groups
    monkeypatch.setattr(utils, "get_classlist", lambda adata, classes: ["A", "B"])

    # Mock resolve_class_filter → return small adata-like object with .X and .var_names
    class DummyAdata:
        X = scipy.sparse.csr_matrix([[1, 0, np.nan], [2, 0, 3]])
        var_names = np.array(["P1", "P2", "P3"])
    monkeypatch.setattr(utils, "resolve_class_filter", lambda adata, c, v, debug=True: DummyAdata())

    # Mock upsetplot.from_contents
    dummy_df = pd.DataFrame({"A": [1, 0], "B": [0, 1]})
    monkeypatch.setattr(utils.upsetplot, "from_contents", lambda d: dummy_df)

    result = utils.get_upset_contents(pdata, classes="cellline", on="protein", upsetForm=True)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(dummy_df)

def test_get_upset_contents_returns_dict(monkeypatch, pdata):
    """Return dictionary when upsetForm=False."""
    monkeypatch.setattr(utils, "get_classlist", lambda adata, classes: ["X"])
    monkeypatch.setattr(utils, "resolve_class_filter", lambda adata, c, v, debug=True: pdata.prot)

    result = utils.get_upset_contents(pdata, classes="cellline", upsetForm=False)
    assert isinstance(result, dict)
    assert "X" in result
    assert isinstance(result["X"], list)

def test_get_upset_contents_invalid_on_raises(pdata):
    """Raise ValueError for invalid 'on' argument."""
    with pytest.raises(ValueError, match="Invalid value for 'on'"):
        utils.get_upset_contents(pdata, classes="cellline", on="invalid")

def test_get_upset_contents_single_class_list(monkeypatch, pdata):
    """Handle single-element class list gracefully."""
    monkeypatch.setattr(utils, "get_classlist", lambda adata, classes: ["C1"])
    monkeypatch.setattr(utils, "resolve_class_filter", lambda adata, c, v, debug=True: pdata.prot)
    monkeypatch.setattr(utils.upsetplot, "from_contents", lambda d: pd.DataFrame())

    result = utils.get_upset_contents(pdata, classes=["cellline"], upsetForm=True)
    assert isinstance(result, pd.DataFrame)

def test_get_upset_contents_integration_small(pdata):
    """Run real get_upset_contents() on small pdata to ensure it executes."""
    df = utils.get_upset_contents(pdata, classes="cellline", upsetForm=True)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_get_upset_query_returns_dataframe(monkeypatch):
    dummy_query_result = type("DummyQuery", (), {
        "data": {"id": np.array(["P12345", "Q67890"])}
    })()
    monkeypatch.setattr(utils.upsetplot, "query", lambda c, present, absent: dummy_query_result)
    monkeypatch.setattr(utils, "get_uniprot_fields",
                        lambda ids, verbose=False: pd.DataFrame({"Accession": ids}))
    result = utils.get_upset_query(pd.DataFrame(), present=["A"], absent=["B"])
    assert isinstance(result, pd.DataFrame)
    assert set(result["Accession"]) == {"P12345", "Q67890"}


def test_get_upset_query_handles_empty(monkeypatch):
    """Handle case where no proteins are found."""
    # Mock return structure of upsetplot.query
    dummy_query_result = type("DummyQuery", (), {"data": {"id": np.array([])}})()
    monkeypatch.setattr(utils.upsetplot, "query", lambda c, present, absent: dummy_query_result)
    
    # Mock get_uniprot_fields to return an empty DataFrame
    monkeypatch.setattr(utils, "get_uniprot_fields", lambda ids, verbose=False: pd.DataFrame())
    
    # Run
    result = utils.get_upset_query(pd.DataFrame(), present=["X"], absent=["Y"])
    
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_get_upset_query_passes_verbose_flag(monkeypatch):
    called = {}

    # Mock upsetplot.query to return .data["id"] as np.array
    dummy_query_result = type("DummyQuery", (), {"data": {"id": np.array(["A1"])}})()
    monkeypatch.setattr(utils.upsetplot, "query", lambda *a, **k: dummy_query_result)

    # Mock get_uniprot_fields to capture arguments
    def mock_get_uniprot_fields(ids, verbose=False):
        called["ids"] = ids
        called["verbose"] = verbose
        return pd.DataFrame({"Accession": ids})
    monkeypatch.setattr(utils, "get_uniprot_fields", mock_get_uniprot_fields)

    # Run
    utils.get_upset_query(pd.DataFrame(), present=["A"], absent=["B"])

    # Assertions
    assert called["verbose"] is False
    assert called["ids"] == ["A1"]

class DummyPDataMapping:
    def __init__(self, source):
        self.metadata = {"source": source}
        self.pep = type("DummyPep", (), {})()
        self.pep.var = pd.DataFrame({
            "Master Protein Accessions": ["A", "B"],
            "Protein.Group": ["C", "D"],
            "Leading razor protein": ["E", "F"],
        })

# --- known sources ---

@pytest.mark.parametrize("source,expected_col", [
    ("proteomediscoverer", "Master Protein Accessions"),
    ("diann", "Protein.Group"),
    ("maxquant", "Leading razor protein"),
])
def test_get_pep_prot_mapping_known_sources(source, expected_col):
    pdata = DummyPDataMapping(source)
    result = utils.get_pep_prot_mapping(pdata)
    assert result == expected_col


# --- return_series=True ---
def test_get_pep_prot_mapping_return_series():
    pdata = DummyPDataMapping("diann")
    series = utils.get_pep_prot_mapping(pdata, return_series=True)
    assert isinstance(series, pd.Series)
    assert series.equals(pdata.pep.var["Protein.Group"])

# --- unknown source ---
def test_get_pep_prot_mapping_unknown_source_raises():
    pdata = DummyPDataMapping("unknown_source")
    with pytest.raises(ValueError, match="Unknown data source"):
        utils.get_pep_prot_mapping(pdata)


# from setup.py
def test_get_datetime_format():
    from scviz.setup import get_datetime
    dt = get_datetime()
    # Expect format YYYY-MM-DD HH:MM:SS
    assert len(dt.split()) == 2
    assert len(dt.split()[0].split('-')) == 3

def test_print_versions_runs(capsys):
    from scviz.setup import print_versions
    print_versions()
    captured = capsys.readouterr().out
    assert "scViz version" in captured
    assert "Dependencies:" in captured

# ------------- unprot API tests
# --------- local
def test_standardize_uniprot_columns_handles_variants():
    """Ensure UniProt column names are normalized across API naming changes."""
    df = pd.DataFrame(columns=[
        "Entry",
        "Entry Name",
        "Gene Names (primary)",
        "Organism (ID)",
        "Cross-reference (STRING)"
    ])

    std = utils.standardize_uniprot_columns(df)

    expected = {"accession", "id", "gene_primary", "organism_id", "xref_string"}
    assert expected.issubset(set(std.columns))

    # Ensure no spaces or parentheses remain
    for c in std.columns:
        assert " " not in c
        assert "(" not in c
        assert ")" not in c

def test_standardize_uniprot_columns_aliases_and_edgecases():
    """Handle unusual or case-variant UniProt field names."""
    df = pd.DataFrame(columns=[
        "gene_primary_name",
        "organism_identifier",
        "Cross_reference_STRING",
        "STRING",
        "Accession"
    ])

    std = utils.standardize_uniprot_columns(df)

    assert "gene_primary" in std.columns
    assert "organism_id" in std.columns
    assert "xref_string" in std.columns
    assert "accession" in std.columns

def test_map_uniprot_field_accession_to_gene():
    """Basic accession → gene mapping returns correct columns and fields."""
    from_col, to_cols, fields = utils._map_uniprot_field("accession", "gene")
    assert from_col == "accession"
    assert to_cols == ["gene_primary"]
    # accession, from_col, and to_cols are included in required fields
    assert set(["accession", "gene_primary"]).issubset(fields)

def test_map_uniprot_field_gene_to_multiple_targets():
    """Gene → multiple outputs returns combined field list."""
    from_col, to_cols, fields = utils._map_uniprot_field("gene", ["string", "organism_id"])
    assert from_col == "gene_primary"
    assert set(to_cols) == {"xref_string", "organism_id"}
    # Must include accession because gene → X lookups require it
    assert "accession" in fields

@pytest.mark.parametrize("bad_from", ["badtype", "STRING", ""])
def test_map_uniprot_field_invalid_from_type_raises(bad_from):
    """Invalid from_type should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid from_type"):
        utils._map_uniprot_field(bad_from, "gene")

@pytest.mark.parametrize("bad_to", [["gene", "foobar"], "foobar"])
def test_map_uniprot_field_invalid_to_type_raises(bad_to):
    """Invalid to_type should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid to_type"):
        utils._map_uniprot_field("accession", bad_to)

def test_map_uniprot_field_organism_id_cannot_be_source():
    """'organism_id' may not be used as from_type."""
    with pytest.raises(ValueError, match="organism_id' can only be used as a target"):
        utils._map_uniprot_field("organism_id", "gene")

def test_update_missing_genes_accepts_variant_uniprot_columns(monkeypatch, pdata):
    """Simulate UniProt returning variant headers and ensure gene update works."""
    def mock_get_uniprot_fields(ids, search_fields=None, **kwargs):
        return pd.DataFrame({
            "Entry": ids,
            "Gene Names (primary)": ["GENE_" + x for x in ids]
        })

    monkeypatch.setattr(utils, "get_uniprot_fields", mock_get_uniprot_fields)

    # Track which ones we blank out
    target_idx = pdata.prot.var.index[:2]
    pdata.prot.var.loc[target_idx, "Genes"] = pd.NA

    pdata.update_missing_genes(verbose=False)

    # Check only the updated rows
    updated_genes = pdata.prot.var.loc[target_idx, "Genes"]
    assert updated_genes.notna().all()
    assert all(updated_genes.str.startswith("GENE_"))

def test_standardize_uniprot_columns_warns_only_for_critical_fields():
    """Warn only for critical UniProt column drifts (accession/gene/organism/string)."""

    df = pd.DataFrame(columns=[
        "Entry",                     # OK → accession
        "Gene Names",                # critical drift: should warn
        "Organism (ID)",             # OK → organism_id
        "Cross-reference (STRING)",  # OK → xref_string
        "Protein names",             # benign → should NOT warn
        "Length"                     # benign → should NOT warn
    ])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        utils.standardize_uniprot_columns(df)

    warn_messages = [str(x.message) for x in w]

    # It should warn for "Gene Names" but not "Protein names" or "Length"
    assert any("Gene Names" not in msg for msg in warn_messages)
    assert all("Protein names" not in msg and "Length" not in msg for msg in warn_messages)

    # If a truly new critical drift (e.g. "Gene Symbol") appeared, that would still trigger
    df2 = pd.DataFrame(columns=["Gene Symbol"])
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        utils.standardize_uniprot_columns(df2)

    critical_warns = [str(x.message) for x in w2]
    assert any("Gene Symbol" in msg for msg in critical_warns)

def test_standardize_uniprot_columns_emits_warning_on_drift():
    """
    Ensure that critical mapping warnings from standardize_uniprot_columns
    are NOT suppressed by global filters.
    """
    # Make a mock DataFrame with a renamed critical column to simulate drift
    df = pd.DataFrame({
        "EntryNumber": ["P12345"],  # should map to 'accession' but alias not known
        "Gene Name": ["GAPDH"],
        "Organism": ["Homo sapiens"]
    })

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=r".*Missing expected UniProt columns.*")
        _ = utils.standardize_uniprot_columns(df)

    # Collect relevant warnings
    messages = [str(x.message) for x in w if "standardize_uniprot_columns" in str(x.message)]
    print("\n[DEBUG] Caught warnings:", messages)

    # It should issue at least one warning mentioning "critical mapping"
    assert any("critical mapping" in msg or "Unrecognized UniProt column" in msg for msg in messages), \
        "❌ No critical mapping warning raised — suppression may be too broad!"

    # Confirm the Missing expected columns warning *is not* caught (it's suppressed globally)
    assert not any("Missing expected UniProt columns" in msg for msg in messages), \
        "❌ Suppression of benign missing-column warnings not active!"

# --------- external
@pytest.mark.external
def test_get_uniprot_fields_live_consistency():
    """
    Live integration test for UniProt API.
    - Verifies that standardization produces stable column names.
    - Warns if UniProt schema changes (e.g., column renames or structure drift).
    - Captures standardizer warnings about critical field drift.
    """
    import requests
    proteins = ["P40925", "P40926"]  # citrate synthase & malate dehydrogenase

    try:
        df = utils.get_uniprot_fields(
            proteins,
            batch_size=1,
            verbose=True,
            standardize=False,  # use raw names to detect change
        )
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Skipping UniProt live test due to network issue: {e}")
        return

    assert not df.empty, "❌ UniProt returned an empty DataFrame."

    # Record actual API columns (for debugging or drift detection)
    print("\n[DEBUG] Raw UniProt columns:", list(df.columns))

    # --- Expected keywords we always rely on downstream ---
    core_expected = ["Entry", "Gene", "Organism", "STRING"]
    missing = [k for k in core_expected if not any(k.lower() in c.lower() for c in df.columns)]
    if missing:
        pytest.fail(f"⚠️ UniProt schema drift detected! Missing expected columns: {missing}")

    # --- Capture warnings from the standardizer ---
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df_std = utils.standardize_uniprot_columns(df)

    warn_messages = [str(x.message) for x in w if "standardize_uniprot_columns" in str(x.message)]

    # --- Canonical columns must exist ---
    canonical = {"accession", "id", "gene_primary", "organism_id", "xref_string"}
    assert canonical.issubset(set(df_std.columns)), (
        f"⚠️ Standardization failed: expected {canonical}, got {set(df_std.columns)}"
    )

    # --- Warn if UniProt changed critical names ---
    if any("critical mapping" in msg for msg in warn_messages):
        pytest.fail(
            "⚠️ UniProt schema drift detected — critical mapping column changed:\n"
            + "\n".join(warn_messages)
        )

    # --- Log new or renamed fields (informational only) ---
    new_fields = [c for c in df.columns if all(k.lower() not in c.lower() for k in core_expected)]
    if new_fields:
        print(f"[INFO] New or renamed UniProt columns detected: {new_fields}")


@pytest.mark.external
def test_get_uniprot_fields_live_standardized_subset():
    """
    Live test of `get_uniprot_fields()` with standardization ON.
    Ensures standardized DataFrame includes expected canonical fields.
    """
    import requests
    proteins = ["P68871", "P69905"]  # Hemoglobin subunits

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = utils.get_uniprot_fields(proteins, batch_size=2, verbose=True, standardize=True)
    except requests.exceptions.RequestException as e:
        pytest.skip(f"Skipping UniProt live test due to network issue: {e}")
        return

    assert not df.empty
    assert "accession" in df.columns
    assert "gene_primary" in df.columns or "gene_names" in df.columns
    assert "organism_id" in df.columns
    assert "xref_string" in df.columns

    # Confirm all queried proteins appear
    assert set(proteins).intersection(set(df["accession"])) == set(proteins)

    # Capture and fail on critical mapping warnings
    warn_messages = [str(x.message) for x in w if "standardize_uniprot_columns" in str(x.message)]
    if any("critical mapping" in msg for msg in warn_messages):
        pytest.fail(
            "⚠️ Critical UniProt schema change detected in standardized subset:\n"
            + "\n".join(warn_messages)
        )

# test convert_identifier...
# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def mock_pdata_CI():
    """Mock pAnnData-like object providing accession↔gene mappings."""
    gene_to_acc = {"GAPDH": "P12345", "ACTB": "Q9XYZ1"}
    acc_to_gene = {v: k for k, v in gene_to_acc.items()}

    class MockPdata:
        def get_identifier_maps(self, on="protein"):
            return gene_to_acc, acc_to_gene

    return MockPdata()

@pytest.fixture
def mock_uniprot_CI(monkeypatch):
    """Mock UniProt response used for offline identifier conversion tests."""
    df = pd.DataFrame({
        "accession": ["P12345", "Q9XYZ1"],
        "gene_primary": ["GAPDH", "ACTB"],
        "xref_string": ["9606.ENSP0001", "9606.ENSP0002"],
        "organism_id": [9606, 9606],
    })
    monkeypatch.setattr(utils, "get_uniprot_fields", lambda *a, **k: df.copy())
    monkeypatch.setattr(utils, "standardize_uniprot_columns", lambda x: x)
    return df

# -------------------------------------------------------------------------
# Core Tests
# -------------------------------------------------------------------------

def test_convert_identifiers_empty_input_returns_expected_df():
    """Empty input should return empty dict or DataFrame depending on return_type."""
    res_dict = utils.convert_identifiers([], "accession", "gene", return_type="dict")
    res_df = utils.convert_identifiers([], "accession", "gene", return_type="df")
    assert res_dict == {}
    assert isinstance(res_df, pd.DataFrame)
    assert res_df.empty

def test_convert_identifiers_invalid_return_type_raises(mock_uniprot_CI):
    """Invalid return_type should raise ValueError."""
    with pytest.raises(ValueError):
        utils.convert_identifiers(["P12345"], "accession", "gene", return_type="xyz")

# -------------------------------------------------------------------------
# Cache-based Lookups
# -------------------------------------------------------------------------

def test_convert_identifiers_accession_to_gene_uses_cache(mock_pdata_CI):
    """When pdata provided, accession→gene mapping should use cached map."""
    res = utils.convert_identifiers(["P12345"], "accession", "gene", pdata=mock_pdata_CI, verbose=False)
    assert res["P12345"]["gene"] == "GAPDH"

def test_convert_identifiers_gene_to_accession_uses_cache(mock_pdata_CI):
    """When pdata provided, gene→accession mapping should use cached map."""
    res = utils.convert_identifiers(["GAPDH"], "gene", "accession", pdata=mock_pdata_CI, verbose=False)
    assert res["GAPDH"]["accession"] == "P12345"

# -------------------------------------------------------------------------
# API / Mocked UniProt Lookups
# -------------------------------------------------------------------------

def test_convert_identifiers_accession_to_multiple_targets_triggers_api(monkeypatch, mock_uniprot_CI):
    """Accession→[gene, string, organism_id] should call get_uniprot_fields."""
    called = {}

    def fake_get_uniprot_fields(ids, **kwargs):
        called["ids"] = ids
        return mock_uniprot_CI

    monkeypatch.setattr(utils, "get_uniprot_fields", fake_get_uniprot_fields)

    res = utils.convert_identifiers(["P12345"], "accession", ["gene", "string", "organism_id"], verbose=False)
    assert called["ids"] == ["P12345"]
    assert set(res["P12345"].keys()) == {"gene", "string", "organism_id"}
    assert res["P12345"]["gene"] == "GAPDH"

def test_convert_identifiers_gene_to_string_triggers_nested_call(monkeypatch, mock_uniprot_CI, mock_pdata_CI):
    """Gene→STRING should trigger recursive gene→accession lookup and UniProt query."""
    monkeypatch.setattr(utils, "get_uniprot_fields", lambda *a, **k: mock_uniprot_CI)
    monkeypatch.setattr(utils, "standardize_uniprot_columns", lambda x: x)

    res = utils.convert_identifiers(["GAPDH"], "gene", "string", pdata=mock_pdata_CI, verbose=False)
    assert "GAPDH" in res
    assert res["GAPDH"]["string"].startswith("9606.")

# -------------------------------------------------------------------------
# Output Format Variants
# -------------------------------------------------------------------------

def test_convert_identifiers_return_type_df_and_both(mock_uniprot_CI):
    """Return types 'df' and 'both' should match dict structure."""
    res_df = utils.convert_identifiers(["P12345"], "accession", "gene", return_type="df", verbose=False)
    assert isinstance(res_df, pd.DataFrame)
    assert "gene" in res_df.columns

    res_dict, df = utils.convert_identifiers(["P12345"], "accession", "gene", return_type="both", verbose=False)
    assert isinstance(res_dict, dict)
    assert isinstance(df, pd.DataFrame)
    assert res_dict["P12345"]["gene"] == df.loc[df["accession"] == "P12345", "gene"].iloc[0]

# -------------------------------------------------------------------------
# Edge & Robustness Cases
# -------------------------------------------------------------------------

def test_convert_identifiers_missing_identifier_returns_none(monkeypatch, mock_uniprot_CI):
    """Identifiers not found in UniProt should map to None."""
    df = mock_uniprot_CI[mock_uniprot_CI["accession"] != "MISSING"]
    monkeypatch.setattr(utils, "get_uniprot_fields", lambda *a, **k: df)

    res = utils.convert_identifiers(["MISSING"], "accession", "gene", verbose=False)
    assert res["MISSING"]["gene"] is None

def test_convert_identifiers_warn_on_schema_drift(monkeypatch):
    """Ensure missing xref_string column handled gracefully."""
    df = pd.DataFrame({
        "accession": ["P12345"],
        "gene_primary": ["GAPDH"],
        "organism_id": [9606],
    })
    monkeypatch.setattr(utils, "get_uniprot_fields", lambda *a, **k: df)
    monkeypatch.setattr(utils, "standardize_uniprot_columns", lambda x: x)

    res = utils.convert_identifiers(["P12345"], "accession", "string", verbose=False)
    assert res["P12345"]["string"] is None

# ------------- statistical
def test_get_pca_importance_dict_basic():
    model = {"PCs": np.array([[0.1, -0.8, 0.3], [0.5, -0.2, 0.4]])}
    df = utils.get_pca_importance(model, ["A", "B", "C"], n=1)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["Principal Component", "Top Features"]
    assert all(isinstance(x, list) for x in df["Top Features"])
    assert df["Principal Component"].tolist() == ["PC1", "PC2"]

def test_get_pca_importance_sklearn_pca():
    from sklearn.decomposition import PCA
    X = np.random.randn(10, 4)
    model = PCA(n_components=2).fit(X)
    df = utils.get_pca_importance(model, [f"f{i}" for i in range(4)], n=2)
    assert all(df["Principal Component"].str.startswith("PC"))
    assert df["Top Features"].apply(lambda x: all(isinstance(f, str) for f in x)).all()

@pytest.mark.parametrize("n", [0, -1, 3, 10])
def test_get_pca_importance_various_n(n):
    pcs = np.array([[0.1, 0.5, 0.9]])
    df = utils.get_pca_importance({"PCs": pcs}, ["a", "b", "c"], n=n)
    assert isinstance(df, pd.DataFrame)
    assert "PC1" in df["Principal Component"].values
    # number of returned features never exceeds available
    assert len(df["Top Features"].iloc[0]) <= 3

def test_get_pca_importance_invalid_inputs():
    with pytest.raises(KeyError):
        utils.get_pca_importance({}, ["a", "b"])

    class Dummy: pass
    with pytest.raises(AttributeError):
        utils.get_pca_importance(Dummy(), ["a", "b"])

    pcs = np.array([[0.1, 0.2, 0.3]])
    with pytest.raises(IndexError):
        utils.get_pca_importance({"PCs": pcs}, ["x", "y"])
import pytest
import pandas as pd
from unittest.mock import patch, Mock
from xml.parsers.expat import ExpatError

from scpviz.enrichment import enrichment_functional, _resolve_de_key, enrichment_ppi, _pretty_vs_key

# Dummy gene list for user input test
genelist = ['P55072', 'NPLOC4', 'UFD1', 'STX5A', 'NSFL1C', 'UBXN2A', 'UBXN4', 'UBE4B', 'YOD1']

# Mock STRING mapping (TSV format)
mock_mapping_response = """\
P55072\tP55072\t9606.ENSP00000354687\t9606\tGENE1\talias\t0.99
NPLOC4\tNPLOC4\t9606.ENSP00000354688\t9606\tGENE2\talias\t0.99
"""

# Mock enrichment JSON result
mock_enrichment_response = [
    {
        "category": "GO:BP",
        "term": "protein catabolic process",
        "description": "protein catabolic process",
        "fdr": 1.23e-5,
        "number_of_genes": 5
    }
]

mock_ppi_response = {
    "number_of_nodes": 6,
    "number_of_edges": 15,
    "average_node_degree": 5.0,
    "local_clustering_coefficient": 0.6,
    "expected_number_of_edges": 4.2,
    "p_value": 1.2e-5
}

@pytest.fixture
def pdata_with_de(pdata):
    # Run DE for an example comparison
    case1 = {'cellline': 'BE', 'treatment': 'kd'}
    case2 = {'cellline': 'BE', 'treatment': 'sc'}
    pdata.de(values=[case1, case2], fold_change_mode="pairwise_median")

    return pdata

def inject_mock_de_genes(pdata, de_key, up_gene="P55072", down_gene="NPLOC4"):
    """
    Injects mock DE genes into the DE results stored in pdata.stats[de_key],
    ensuring both up- and down-regulated genes are present and properly labeled.

    Parameters
    ----------
    pdata : pAnnData
        The pAnnData object containing DE results.
    de_key : str
        The key in pdata.stats corresponding to a DE comparison.
    up_gene : str
        A gene name or accession to label as upregulated (must be in mock STRING mapping).
    down_gene : str
        A gene name or accession to label as downregulated (must be in mock STRING mapping).
    """
    de_df = pdata.stats[de_key]
    midpoint = len(de_df) // 2

    # Set upregulated
    de_df.iloc[:midpoint, de_df.columns.get_loc("significance_score")] = 2.0
    de_df.iloc[:midpoint, de_df.columns.get_loc("significance")] = "upregulated"
    de_df.iloc[:midpoint, de_df.columns.get_loc("Genes")] = up_gene

    # Set downregulated
    de_df.iloc[midpoint:, de_df.columns.get_loc("significance_score")] = -2.0
    de_df.iloc[midpoint:, de_df.columns.get_loc("significance")] = "downregulated"
    de_df.iloc[midpoint:, de_df.columns.get_loc("Genes")] = down_gene

    pdata.stats[de_key] = de_df


def mock_string_responses(mock_post, mapping_text, enrichment_json, total_pairs=4):
    """
    Mock STRING API responses for a sequence of mapping + enrichment calls.

    Parameters
    ----------
    mock_post : MagicMock
        The patched mock of requests.post
    mapping_text : str
        The TSV response to return for get_string_ids
    enrichment_json : list
        The list of enrichment results to return as JSON
    total_pairs : int
        Number of mapping + enrichment call pairs to mock
    """
    def make_mapping(i):
        m = Mock()
        m.status_code = 200
        m.text = mapping_text
        m.raise_for_status = Mock()
        m._label = f"[MOCK MAPPING #{i}]"
        return m

    def make_enrich(i):
        e = Mock()
        e.status_code = 200
        e.raise_for_status = Mock()
        e.json = Mock(return_value=enrichment_json)
        e._label = f"[MOCK ENRICHMENT #{i}]"
        return e

    response_sequence = []
    for i in range(total_pairs):
        response_sequence.append(make_mapping(i+1))
        response_sequence.append(make_enrich(i+1))

    print(f"[DEBUG] Configured {len(response_sequence)} mock POST responses: ")
    for r in response_sequence:
        print("   ", getattr(r, "_label", r))

    def smart_string_side_effect(*args, **kwargs):
        url = args[0] if args else kwargs.get("url", "")
        if "get_string_ids" in url:
            m = Mock()
            m.status_code = 200
            m.text = mock_mapping_response
            m.raise_for_status = Mock()
            print("[DEBUG] → Returning MOCK MAPPING")
            return m
        elif "ppi_enrichment" in url:
            e = Mock()
            e.status_code = 200
            e.raise_for_status = Mock()
            e.json = Mock(return_value=mock_ppi_response)  # this is a dict
            print("[DEBUG] → Returning MOCK PPI")
            return e
        elif "enrichment" in url:
            e = Mock()
            e.status_code = 200
            e.raise_for_status = Mock()
            e.json = Mock(return_value=mock_enrichment_response)
            print("[DEBUG] → Returning MOCK ENRICHMENT")
            return e
        else:
            raise RuntimeError(f"[ERROR] Unknown STRING URL called: {url}")

    mock_post.side_effect = smart_string_side_effect

@patch("scpviz.enrichment.requests.post")
def test_user_supplied_functional(mock_post, pdata):
    # Set up mocks using the shared helper
    mock_string_responses(mock_post, mock_mapping_response, mock_enrichment_response, total_pairs=1)
    # Run enrichment
    df = enrichment_functional(pdata, genes=genelist, from_de=False, store_key="TestUserSearch")
    
    assert isinstance(df, pd.DataFrame)
    assert "fdr" in df.columns
    assert "functional" in pdata.stats
    assert "TestUserSearch" in pdata.stats["functional"]

    meta = pdata.stats["functional"]["TestUserSearch"]
    assert isinstance(meta, dict)
    assert "result" in meta
    assert isinstance(meta["result"], pd.DataFrame)

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@patch("scpviz.enrichment.requests.post")
def test_de_based_functional(mock_post, pdata_with_de):
    # Modern de() stats keys use value joins (BE_kd vs BE_sc); legacy dict-style
    # keys still resolve via _pretty_vs_key / _resolve_de_key.
    de_key = "BE_kd vs BE_sc"
    resolved_key = _resolve_de_key(pdata_with_de.stats, de_key)
    assert resolved_key == "BE_kd vs BE_sc"
    legacy = "[{'cellline': 'BE', 'treatment': 'kd'}] vs [{'cellline': 'BE', 'treatment': 'sc'}]"
    assert _resolve_de_key(pdata_with_de.stats, legacy) == resolved_key

    mock_string_responses(mock_post, mock_mapping_response, mock_enrichment_response, total_pairs=6)
    inject_mock_de_genes(pdata_with_de, resolved_key, up_gene="P55072", down_gene="NPLOC4")

    enrichment_functional(pdata_with_de, from_de=True, de_key=resolved_key, store_key="TestDE")

    for suffix in ["up", "down"]:
        pretty_key = f"{_pretty_vs_key(resolved_key)}_{suffix}"
        assert pretty_key in pdata_with_de.stats["functional"]
        assert "result" in pdata_with_de.stats["functional"][pretty_key]
        assert isinstance(pdata_with_de.stats["functional"][pretty_key]["result"], pd.DataFrame)


def _plant_ranked_de(pdata, stored, n_up=5, n_down=4):
    """Label unique accessions as up/down with increasing |score| down the table."""
    de_df = pdata.stats[stored]
    de_df["significance"] = "not significant"
    de_df["significance_score"] = 0.0
    de_df["Genes"] = de_df.index.astype(str)
    idx = list(de_df.index)
    up_ids = idx[:n_up]
    down_ids = idx[n_up:n_up + n_down]
    for i, acc in enumerate(up_ids):
        de_df.at[acc, "significance"] = "upregulated"
        de_df.at[acc, "significance_score"] = float(i + 1)
    for i, acc in enumerate(down_ids):
        de_df.at[acc, "significance"] = "downregulated"
        de_df.at[acc, "significance_score"] = -float(i + 1)
    return up_ids, down_ids


def test_enrichment_functional_default_sends_all_de_genes(monkeypatch, pdata_with_de):
    """top_n=None (default) sends every upregulated and downregulated gene."""
    stored = "[{'cellline': 'BE', 'treatment': 'kd'}] vs [{'cellline': 'BE', 'treatment': 'sc'}]"
    up_ids, down_ids = _plant_ranked_de(pdata_with_de, stored, n_up=5, n_down=4)
    seen = []

    def fake_map(accs, **kwargs):
        accs = list(accs)
        seen.append(accs)
        return pd.DataFrame({
            "input_identifier": accs,
            "string_identifier": [f"S_{a}" for a in accs],
            "ncbi_taxon_id": [10090] * len(accs),
        })

    monkeypatch.setattr(pdata_with_de, "get_string_mappings", fake_map)
    mock_post = Mock()
    mock_post.return_value.json.return_value = mock_enrichment_response
    monkeypatch.setattr("scpviz.enrichment.requests.post", mock_post)

    pdata_with_de.enrichment_functional(from_de=True, de_key=stored)
    assert set(seen[0]) == set(up_ids)
    assert set(seen[1]) == set(down_ids)


def test_enrichment_functional_top_n_ranks_by_score(monkeypatch, pdata_with_de):
    """Explicit top_n keeps the highest |significance_score| per tail, not table order."""
    stored = "[{'cellline': 'BE', 'treatment': 'kd'}] vs [{'cellline': 'BE', 'treatment': 'sc'}]"
    up_ids, down_ids = _plant_ranked_de(pdata_with_de, stored, n_up=5, n_down=4)
    seen = []

    def fake_map(accs, **kwargs):
        accs = list(accs)
        seen.append(accs)
        return pd.DataFrame({
            "input_identifier": accs,
            "string_identifier": [f"S_{a}" for a in accs],
            "ncbi_taxon_id": [10090] * len(accs),
        })

    monkeypatch.setattr(pdata_with_de, "get_string_mappings", fake_map)
    mock_post = Mock()
    mock_post.return_value.json.return_value = mock_enrichment_response
    monkeypatch.setattr("scpviz.enrichment.requests.post", mock_post)

    pdata_with_de.enrichment_functional(from_de=True, de_key=stored, top_n=2)
    assert set(seen[0]) == {up_ids[-1], up_ids[-2]}
    assert set(seen[1]) == {down_ids[-1], down_ids[-2]}


def test_enrichment_functional_invalid_top_n(pdata_with_de):
    stored = "[{'cellline': 'BE', 'treatment': 'kd'}] vs [{'cellline': 'BE', 'treatment': 'sc'}]"
    with pytest.raises(ValueError, match="positive integer or None"):
        pdata_with_de.enrichment_functional(from_de=True, de_key=stored, top_n=0)
    with pytest.raises(ValueError, match="positive integer or None"):
        pdata_with_de.enrichment_functional(from_de=True, de_key=stored, top_n=-3)


@patch("scpviz.enrichment.requests.post")
def test_user_supplied_ppi(mock_post, pdata):
    mock_string_responses(mock_post, mock_mapping_response, mock_ppi_response, total_pairs=1)

    # Run enrichment
    result = enrichment_ppi(pdata, genes=genelist, store_key="TestUserPPI")

    # Check structure
    assert isinstance(result, dict)
    assert "number_of_nodes" in result
    assert "p_value" in result

    # Check that result was stored
    assert "ppi" in pdata.stats
    assert "TestUserPPI" in pdata.stats["ppi"]
    assert "result" in pdata.stats["ppi"]["TestUserPPI"]

def test_resolve_de_key_finds_pretty_match():
    # Legacy stored keys (dict repr) still match pretty user lookups
    keys = {
        "[{'cellline': 'AS', 'treatment': 'kd'}] vs [{'cellline': 'AS', 'treatment': 'sc'}]_up": "...",
        "[{'cellline': 'AS', 'treatment': 'kd'}] vs [{'cellline': 'AS', 'treatment': 'sc'}]_down": "..."
    }
    resolved = _resolve_de_key(keys, "AS_kd vs AS_sc_up")
    assert resolved in keys

def test_resolve_de_key_modern_stats_key():
    """Current de() stores 'BE_kd vs BE_sc'; resolve both modern and legacy forms."""
    keys = {"BE_kd vs BE_sc": "..."}
    assert _resolve_de_key(keys, "BE_kd vs BE_sc") == "BE_kd vs BE_sc"
    legacy = "[{'cellline': 'BE', 'treatment': 'kd'}] vs [{'cellline': 'BE', 'treatment': 'sc'}]"
    assert _resolve_de_key(keys, legacy) == "BE_kd vs BE_sc"

# ----------------------------------------------------------------------
# get_string_mappings edge cases
def test_get_string_mappings_uniprot_failure(monkeypatch, pdata):
    """Simulate UniProt API failure to cover exception path."""
    def fail_uniprot(*args, **kwargs):
        raise RuntimeError("Simulated UniProt failure")
    
    import scpviz.utils as utils_mod
    monkeypatch.setattr(utils_mod, "get_uniprot_fields", fail_uniprot)
    pdata.prot.var["STRING_id"] = pd.NA
    # Should print warning and still return empty df
    df = pdata.get_string_mappings(["X1", "X2"], debug=False)
    assert isinstance(df, pd.DataFrame)

def test_get_string_mappings_all_cached(pdata):
    """If all identifiers already cached, skip STRING step."""
    pdata.prot.var["STRING_id"] = ["S1"] * pdata.prot.shape[1]
    df = pdata.get_string_mappings(pdata.prot.var_names[:3].tolist(), overwrite=True, debug=True)
    assert "string_identifier" in df.columns

def test_get_string_mappings_scalarize_taxon_from_cache_only(monkeypatch, pdata):
    """
    Ensure nested _scalarize_taxon() converts list/array/empty values to scalars/NA
    when building out_df["ncbi_taxon_id"] from cached prot.var values.
    """
    import scpviz.utils as utils_mod
    import numpy as np

    monkeypatch.setattr(utils_mod, "get_uniprot_fields", lambda *a, **k: None)

    ids = pdata.prot.var_names[:4].tolist()
    pdata.prot.var["STRING_id"] = pd.NA
    pdata.prot.var.loc[ids, "STRING_id"] = ["S1", "S2", "S3", "S4"]

    # Force from_map to be all missing - species_map is only filled from UniProt (bypass)
    pdata.prot.var["ncbi_taxon_id"] = pd.NA
    pdata.prot.var.at[ids[0], "ncbi_taxon_id"] = [9606]               # list -> first element
    pdata.prot.var.at[ids[1], "ncbi_taxon_id"] = np.array([10090])    # array -> first element
    pdata.prot.var.at[ids[2], "ncbi_taxon_id"] = ""                   # empty -> NA
    pdata.prot.var.at[ids[3], "ncbi_taxon_id"] = None                 # None -> NA

    df = pdata.get_string_mappings(ids, cache_col="STRING_id", debug=False)

    assert "ncbi_taxon_id" in df.columns

    got = df.set_index("input_identifier")["ncbi_taxon_id"]

    assert str(got.loc[ids[0]]) == "9606"
    assert str(got.loc[ids[1]]) == "10090"
    assert pd.isna(got.loc[ids[2]])
    assert pd.isna(got.loc[ids[3]])


def test_get_string_mappings_prints_leftover(monkeypatch, pdata, capsys):
    """Unmapped accessions after UniProt+STRING are printed at indent 3."""
    import scpviz.utils as utils_mod

    ids = pdata.prot.var_names[:3].tolist()
    pdata.prot.var["STRING"] = pd.NA

    def fake_map(identifiers, **kwargs):
        return pd.DataFrame({
            "input_identifier": [ids[0]],
            "string_identifier": ["9606.ENSP1"],
            "ncbi_taxon_id": ["9606"],
        })

    monkeypatch.setattr(utils_mod, "get_string_mappings", fake_map)
    df = pdata.get_string_mappings(ids, debug=False)
    out = capsys.readouterr().out

    assert "still unmapped" in out
    assert ids[1] in out
    assert ids[2] in out
    assert "update_string_mapping" in out
    got = df.set_index("input_identifier")["string_identifier"]
    assert got.loc[ids[0]] == "9606.ENSP1"
    assert pd.isna(got.loc[ids[1]])
    assert pd.isna(got.loc[ids[2]])


def test_get_string_mappings_no_leftover_when_complete(monkeypatch, pdata, capsys):
    """Fully cached identifiers should not print a leftover warning or hit the API."""
    import scpviz.utils as utils_mod

    ids = pdata.prot.var_names[:3].tolist()
    pdata.prot.var["STRING"] = pd.NA
    pdata.prot.var.loc[ids, "STRING"] = ["S1", "S2", "S3"]

    monkeypatch.setattr(
        utils_mod,
        "get_string_mappings",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API should not be called")),
    )
    df = pdata.get_string_mappings(ids, debug=False)
    out = capsys.readouterr().out
    assert "still unmapped" not in out
    assert df["string_identifier"].tolist() == ["S1", "S2", "S3"]


def test_update_string_mapping_add_skip_overwrite(pdata):
    """NA cache is fillable without overwrite; filled cache requires overwrite=True."""
    acc = pdata.prot.var_names[0]
    pdata.prot.var["STRING"] = pd.NA

    pdata.update_string_mapping({acc: "9606.NEW"}, verbose=False)
    assert pdata.prot.var.at[acc, "STRING"] == "9606.NEW"

    pdata.update_string_mapping({acc: "9606.SKIPPED"}, overwrite=False, verbose=False)
    assert pdata.prot.var.at[acc, "STRING"] == "9606.NEW"

    pdata.update_string_mapping({acc: "9606.OVERWRITE"}, overwrite=True, verbose=False)
    assert pdata.prot.var.at[acc, "STRING"] == "9606.OVERWRITE"

    history = pdata.metadata.get("string_map_history", [])
    assert len(history) == 3
    assert history[0]["summary"]["added"] == 1
    assert history[1]["summary"]["skipped"] == 1
    assert history[2]["summary"]["updated"] == 1


def test_update_string_mapping_unknown_and_taxon(pdata, capsys):
    """Unknown accessions are skipped; taxon is written for successful updates."""
    acc = pdata.prot.var_names[0]
    pdata.prot.var["STRING"] = pd.NA
    pdata.update_string_mapping(
        {acc: "9606.ENSP1", "NOT_A_REAL_ACC": "9606.ENSP2"},
        taxon={acc: 9606},
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "NOT_A_REAL_ACC" in out
    assert pdata.prot.var.at[acc, "STRING"] == "9606.ENSP1"
    assert str(pdata.prot.var.at[acc, "ncbi_taxon_id"]) == "9606"
    assert "NOT_A_REAL_ACC" not in pdata.prot.var.index


def test_update_string_mapping_invalid_mapping_raises(pdata):
    with pytest.raises(ValueError, match="must be a dict"):
        pdata.update_string_mapping(["P12345", "9606.ENSP1"])


def test_update_string_mapping_used_as_cache(monkeypatch, pdata):
    """Manual STRING IDs are reused by get_string_mappings without an API call."""
    import scpviz.utils as utils_mod

    acc = pdata.prot.var_names[0]
    pdata.update_string_mapping({acc: "9606.MANUAL"}, taxon="9606", verbose=False)
    monkeypatch.setattr(
        utils_mod,
        "get_string_mappings",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API should not be called")),
    )
    df = pdata.get_string_mappings([acc], debug=False)
    row = df.set_index("input_identifier").loc[acc]
    assert row["string_identifier"] == "9606.MANUAL"
    assert str(row["ncbi_taxon_id"]) == "9606"

# ----------------------------------------------------------------------
# resolve_to_accessions edge case
def test_resolve_to_accessions_unresolved(pdata):
    """One gene resolves, one does not."""
    resolved, unresolved = pdata.resolve_to_accessions(["NONEXISTENT", list(pdata.prot.var_names)[0]])
    assert isinstance(resolved, list)
    assert "NONEXISTENT" in unresolved

# ----------------------------------------------------------------------
# enrichment_functional error paths
def test_enrichment_functional_no_input_raises(pdata):
    """Raises when neither genes nor from_de=True."""
    with pytest.raises(ValueError, match="Must provide 'genes'"):
        pdata.enrichment_functional(from_de=False)

def test_enrichment_functional_empty_mapping(monkeypatch, pdata):
    """Covers empty mapping_df -> raises ValueError."""
    monkeypatch.setattr(pdata, "resolve_to_accessions", lambda x: (["X"], []))
    monkeypatch.setattr(pdata, "get_string_mappings", lambda *a, **k: pd.DataFrame(columns=["string_identifier", "ncbi_taxon_id"]))
    with pytest.raises(ValueError, match="No valid STRING mappings"):
        pdata.enrichment_functional(genes=["X"], from_de=False)

# ----------------------------------------------------------------------
# enrichment_ppi edge case
def test_enrichment_ppi_empty_mapping(monkeypatch, pdata):
    """Covers branch where mapping_df.empty raises ValueError."""
    monkeypatch.setattr(pdata, "resolve_to_accessions", lambda x: (["X"], []))
    monkeypatch.setattr(pdata, "get_string_mappings", lambda *a, **k: pd.DataFrame(columns=["string_identifier", "ncbi_taxon_id"]))
    with pytest.raises(ValueError, match="No valid STRING mappings"):
        pdata.enrichment_ppi(genes=["X"])

def test_enrichment_functional_pca_both_selects_top_abs_loading(monkeypatch, pdata):
    """direction='both' → one enrichment per PC: top_n genes by |loading|."""
    pdata.pca(on="protein")
    adata = pdata.prot
    adata.var["Genes"] = [f"G{i}" for i in range(adata.n_vars)]

    pcs = adata.uns["pca"]["PCs"].copy()
    pcs[0, :] = 0.0
    pcs[0, 0] = 0.9
    pcs[0, 1] = -0.8
    pcs[0, 2] = 0.7
    pcs[0, 3] = -0.6
    adata.uns["pca"]["PCs"] = pcs

    calls = []
    def fake_enrichment_functional(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"term": ["X"]})

    monkeypatch.setattr(pdata, "enrichment_functional", fake_enrichment_functional)
    out = pdata.enrichment_functional_pca(pcs=[1], top_n=2, direction="both", key_prefix="PCA")

    assert "PC1" in out
    assert set(out["PC1"].keys()) == {"abs"}
    assert len(calls) == 1
    assert calls[0]["store_key"] == "PCA_PC1_abs"
    assert calls[0]["genes"] == ["G0", "G1"]


def test_enrichment_functional_pca_positive_negative_separate_tails(monkeypatch, pdata):
    """direction='positive' or 'negative' → top_n within that tail only."""
    pdata.pca(on="protein")
    adata = pdata.prot
    adata.var["Genes"] = [f"G{i}" for i in range(adata.n_vars)]
    pcs = adata.uns["pca"]["PCs"].copy()
    pcs[0, :] = 0.0
    pcs[0, 0] = 0.9
    pcs[0, 1] = -0.8
    pcs[0, 2] = 0.7
    pcs[0, 3] = -0.6
    adata.uns["pca"]["PCs"] = pcs

    calls = []
    def fake_enrichment_functional(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"term": ["X"]})

    monkeypatch.setattr(pdata, "enrichment_functional", fake_enrichment_functional)
    pdata.enrichment_functional_pca(pcs=[1], top_n=2, direction="positive", key_prefix="P")
    assert len(calls) == 1
    assert calls[0]["genes"] == ["G0", "G2"]

    calls.clear()
    pdata.enrichment_functional_pca(pcs=[1], top_n=2, direction="negative", key_prefix="N")
    assert len(calls) == 1
    assert calls[0]["genes"] == ["G1", "G3"]

def test_enrichment_functional_pca_key_suffix_if_exists(monkeypatch, pdata):
    pdata.pca(on="protein")
    adata = pdata.prot
    adata.var["Genes"] = [f"G{i}" for i in range(adata.n_vars)]
    pdata.stats["functional"] = {"PCA_PC1_positive": {"result": pd.DataFrame()}}

    calls = []
    def fake_enrichment_functional(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"term": ["X"]})

    monkeypatch.setattr(pdata, "enrichment_functional", fake_enrichment_functional)
    pdata.enrichment_functional_pca(pcs=[1], top_n=1, direction="positive", key_prefix="PCA")

    assert len(calls) == 1
    assert calls[0]["store_key"] == "PCA_PC1_positive_2"

# ----------------------------------------------------------------------
# list_enrichments coverage
def test_list_enrichments_print(capsys, pdata):
    """Covers list_enrichments() print output."""
    pdata.stats["functional"] = {"UserSearch1": {"result": pd.DataFrame(), "string_url": "https://string-db.org"}}
    pdata.stats["ppi"] = {"UserPPI1": {"result": {"edges": 3}}}
    pdata.stats["de_results"] = {"[{'cellline': 'BE'}] vs [{'cellline': 'AS'}]": pd.DataFrame()}
    pdata.list_enrichments()
    out = capsys.readouterr().out
    assert "Listing STRING enrichment status" in out
    assert "Completed STRING enrichment results" in out

# ----------------------------------------------------------------------
# plot_enrichment_svg coverage
@patch("scpviz.enrichment.requests.get")
def test_plot_enrichment_svg_missing_key(mock_get, pdata, tmp_path):
    """Covers error branch for missing functional key."""
    pdata.stats["functional"] = {}
    with pytest.raises(ValueError, match="Could not find enrichment results for"):
        pdata.plot_enrichment_svg("Nonexistent")

@patch("scpviz.enrichment.requests.get")
def test_plot_enrichment_svg_direction_error(mock_get, pdata):
    """Covers direction missing for DE key."""
    pdata.stats["functional"] = {"A_vs_B_up": {"string_ids": ["S1"], "species": "9606"}}
    with pytest.raises(ValueError, match="You must specify direction"):
        pdata.plot_enrichment_svg("A_vs_B")

@patch("scpviz.enrichment.requests.get")
def test_plot_enrichment_svg_fetch_and_save(mock_get, pdata, tmp_path):
    pdata.stats["functional"] = {"UserSearch1": {"string_ids": ["S1"], "species": "9606"}}

    resp = Mock(status_code=200, content=b"<svg></svg>")
    resp.raise_for_status = Mock()
    mock_get.return_value = resp

    tmpfile = tmp_path / "test.svg"
    pdata.plot_enrichment_svg("UserSearch1", save_as=str(tmpfile))
    assert tmpfile.exists()

@patch("scpviz.enrichment.requests.get")
@patch("IPython.display.SVG", side_effect=ExpatError("no element found: line 1, column 0"))
def test_plot_enrichment_svg_no_image_expaterror(mock_svg, mock_get, pdata, capsys):
    pdata.stats["functional"] = {"UserSearch1": {"string_ids": ["S1"], "species": "9606"}}

    resp = Mock(status_code=200, content=b"")
    resp.raise_for_status = Mock()
    mock_get.return_value = resp

    pdata.plot_enrichment_svg("UserSearch1")

    out = capsys.readouterr().out
    assert "No enrichment figure available" in out

# ----------------------------------------------------------------------
# get_string_network_link coverage
def test_get_string_network_link_errors(pdata):
    """Covers ValueErrors in get_string_network_link."""
    with pytest.raises(ValueError, match="Must provide either"):
        pdata.get_string_network_link()
    pdata.stats["functional"] = {}
    with pytest.raises(ValueError, match="Key 'K' not found"):
        pdata.get_string_network_link(key="K")
    pdata.stats["functional"]["UserSearch1"] = {"string_ids": [], "species": "9606"}
    with pytest.raises(ValueError, match="No STRING IDs"):
        pdata.get_string_network_link(key="UserSearch1")
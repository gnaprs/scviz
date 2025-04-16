import pytest
import pandas as pd
from unittest.mock import patch
from unittest.mock import Mock

import warnings

from scviz.enrichment import enrichment_functional, _resolve_de_key, enrichment_ppi, _pretty_vs_key

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

@patch("scviz.enrichment.requests.post")
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
@patch("scviz.enrichment.requests.post")
def test_de_based_functional(mock_post, pdata_with_de):
    de_key = "[{'cellline': 'BE', 'treatment': 'kd'}] vs [{'cellline': 'BE', 'treatment': 'sc'}]"
    resolved_key = _resolve_de_key(pdata_with_de.stats, de_key)

    mock_string_responses(mock_post, mock_mapping_response, mock_enrichment_response, total_pairs=6)
    inject_mock_de_genes(pdata_with_de, resolved_key, up_gene="P55072", down_gene="NPLOC4")

    enrichment_functional(pdata_with_de, from_de=True, de_key=resolved_key, store_key="TestDE")

    for suffix in ["up", "down"]:
        pretty_key = f"{_pretty_vs_key(resolved_key)}_{suffix}"
        assert pretty_key in pdata_with_de.stats["functional"]
        assert "result" in pdata_with_de.stats["functional"][pretty_key]
        assert isinstance(pdata_with_de.stats["functional"][pretty_key]["result"], pd.DataFrame)

@patch("scviz.enrichment.requests.post")
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
    keys = {
        "[{'cellline': 'AS', 'treatment': 'kd'}] vs [{'cellline': 'AS', 'treatment': 'sc'}]_up": "...",
        "[{'cellline': 'AS', 'treatment': 'kd'}] vs [{'cellline': 'AS', 'treatment': 'sc'}]_down": "..."
    }
    resolved = _resolve_de_key(keys, "AS_kd vs AS_sc_up")
    assert resolved in keys
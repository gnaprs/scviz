import pytest
import pandas as pd
from scpviz import pAnnData
from pathlib import Path

from scpviz.pAnnData.io import _safe_strip

def test_safe_strip_dataframe_and_series():
    # --- DataFrame case ---
    df = pd.DataFrame({"A": [" a", "b ", None], "B": [" c ", " d", "e "]})
    out_df = _safe_strip(df)

    # Verify type and values
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.iloc[0, 0] == "a"
    assert out_df.iloc[1, 1] == "d"
    assert out_df.isna().sum().sum() == 1  # preserves None/NaN

    # --- Series case ---
    s = pd.Series([" x ", "y", None])
    out_s = _safe_strip(s)

    assert isinstance(out_s, pd.Series)
    assert list(out_s) == ["x", "y", None]

    # --- Mixed types should remain unchanged ---
    s_mixed = pd.Series([" a", 5, None])
    out_mixed = _safe_strip(s_mixed)
    assert out_mixed.iloc[1] == 5
    assert out_mixed.iloc[0] == "a"

    # --- Should not raise on empty input ---
    empty_df = pd.DataFrame(columns=["A"])
    _ = _safe_strip(empty_df)

def test_import_pd():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot.txt')
    pep_file = str(test_dir / 'test_pd_pep.txt')

    obs_columns = ['sample', 'cellline', 'treatment']
    pdata = pAnnData.import_data(source_type='pd', prot_file=prot_file, pep_file=pep_file, obs_columns=obs_columns)
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is not None
    assert pdata.rs is not None

def test_import_pd_noQ():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot-noQ.txt')

    obs_columns = ['sample', 'cellline', 'treatment']    
    with pytest.warns(UserWarning, match="Neither 'Exp. q-value: Combined' nor 'Exp. Protein Group q-value: Combined'"):
        pdata = pAnnData.import_data(source_type='pd', prot_file=prot_file, obs_columns=obs_columns)

    # Object should still be valid
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is None
    assert pdata.rs is None

def test_import_pd32():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd32_Proteins.txt')

    obs_columns = ['Sample','cellline','ko','condition','duration']
    pdata = pAnnData.import_data(source_type='pd', prot_file=prot_file, obs_columns=obs_columns)
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is None
    assert pdata.rs is None

def test_import_pd32_with_pep():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd32_Proteins.txt')
    pep_file = str(test_dir / 'test_pd32_PeptideSequenceGroups.txt')

    obs_columns = ['Sample','cellline','ko','condition','duration']
    pdata = pAnnData.import_data(source_type='pd', prot_file=prot_file, pep_file=pep_file, obs_columns=obs_columns)
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is not None
    assert pdata.rs is not None

def test_import_diann_old():
    # pre diann v1.8.1
    test_dir = Path(__file__).parent
    diann_file = str(test_dir / 'test_diann.tsv')

    obs_columns = ['name','amt','enzyme','date','MS','acquisition','method','gradient','replicate']
    pdata = pAnnData.import_data(source_type='diann', report_file=diann_file, obs_columns=obs_columns)
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is not None
    assert pdata.rs is not None

def test_import_diann_new():
    # post diann v1.8.1
    test_dir = Path(__file__).parent
    diann_file = str(test_dir / 'test_diann.parquet')

    # MP_20250219_OA_DIA_FAIMS_TS25_24min_sc_LCM-Cortex_01.raw
    obs_columns = ['name','date','MS','acquisition','FAIMS','column','gradient','amt','region','replicate']
    pdata = pAnnData.import_data(source_type='diann', report_file=diann_file, obs_columns=obs_columns)
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is not None
    assert pdata.rs is not None

def test_import_no_obs_uniform():
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot.txt')

    no_obs_return = pAnnData.import_data(source_type='pd', prot_file=prot_file)
    
    assert no_obs_return is None

def test_import_no_obs_nonuniform():
    test_dir = Path(__file__).parent
    diann_file = str(test_dir / 'test_diann_nonuniform.tsv')

    pdata = pAnnData.import_data(source_type='diann', report_file=diann_file)
    summary = pdata.summary

    assert pdata is not None
    assert 'parsingType' in summary.columns

@pytest.mark.parametrize("on,direction,overwrite", [
    ('protein', 'forward', False),
    ('protein', 'forward', True),
    ('protein', 'reverse', False),
    ('protein', 'reverse', True),
    ('peptide', 'forward', False),
    ('peptide', 'forward', True),
    ('peptide', 'reverse', False),
    ('peptide', 'reverse', True),
])
def test_update_identifier_maps_parametrized(pdata, on, direction, overwrite):
    # Get the current maps
    fwd_map, rev_map = pdata.get_identifier_maps(on=on)

    # Choose dummy test keys depending on direction
    if direction == 'forward':
        test_input = {'TEST_KEY_A': 'TEST_VAL_A'}
        pre_existing_key = 'TEST_KEY_A'
        initial_val = 'OLD_VAL'
        updated_val = 'TEST_VAL_A'
        
        # Inject OLD_VAL into fwd_map[TEST_KEY_A]
        fwd_map[pre_existing_key] = initial_val
        rev_map[initial_val] = pre_existing_key

    else:  # direction == 'reverse'
        test_input = {'TEST_VAL_B': 'TEST_KEY_B'}
        pre_existing_key = 'TEST_VAL_B'
        initial_val = 'OLD_VAL'
        updated_val = 'TEST_KEY_B'
        
        # Inject OLD_VAL into rev_map[TEST_VAL_B]
        rev_map[pre_existing_key] = initial_val
        fwd_map[initial_val] = pre_existing_key


    # Inject a pre-existing key to test overwrite behavior
    fwd_map[pre_existing_key] = initial_val if direction == 'forward' else test_input[pre_existing_key]
    rev_map[initial_val] = pre_existing_key if direction == 'forward' else test_input[pre_existing_key]

    # Run the update
    pdata.update_identifier_maps(test_input, on=on, direction=direction, overwrite=overwrite, verbose=False)

    # Re-fetch the maps
    fwd_map, rev_map = pdata.get_identifier_maps(on=on)

    if overwrite:
        if direction == 'forward':
            assert fwd_map[pre_existing_key] == updated_val
            assert rev_map[updated_val] == pre_existing_key
        else:  # reverse
            assert rev_map[pre_existing_key] == updated_val
            assert fwd_map[updated_val] == pre_existing_key
    else:
        if direction == 'forward':
            assert fwd_map[pre_existing_key] == initial_val
        else:
            assert rev_map[pre_existing_key] == initial_val

    # History message should be added
    history_msgs = getattr(pdata, "_history", [])
    assert any(f"Updated '{on}' ({direction})" in msg for msg in history_msgs)

def test_update_identifier_maps_var_column_sync(pdata):
    # Use the first protein accession in the test data
    acc = pdata.prot.var_names[0]
    old_gene = pdata.prot.var.at[acc, "Genes"]

    # Define a new gene name for that accession
    new_gene = "CUSTOM_GENE_XYZ"
    pdata.update_identifier_maps({acc: new_gene}, on='protein', direction='reverse', overwrite=True, verbose=False)

    # Check that .var["Genes"] was updated
    assert pdata.prot.var.at[acc, "Genes"] == new_gene

    # Check that identifier_map_history was logged
    history = pdata.metadata.get("identifier_map_history", [])
    assert any(
        h.get("on") == "protein" and
        h.get("direction") == "reverse" and
        acc in h.get("updated_var_column", {}).get("accessions", [])
        for h in history
    )

# ----------------------------------------------------------------------
# _build_identifier_maps coverage
def test_build_identifier_maps_protein(pdata):
    """Covers protein mapping branch in _build_identifier_maps."""
    fwd, rev = pdata._build_identifier_maps(pdata.prot, gene_col="Genes")
    # pick a known gene from the dummy file
    gene = list(fwd.keys())[0]
    acc = fwd[gene]
    assert rev[acc] == gene

def test_build_identifier_maps_peptide_warn(monkeypatch, pdata):
    """Forces warning path in _build_identifier_maps (peptide branch exception)."""
    def bad_mapping(_): raise RuntimeError("mock failure")
    monkeypatch.setattr("scpviz.utils.get_pep_prot_mapping", bad_mapping)

    with pytest.warns(UserWarning, match="Could not build peptide-to-protein map"):
        fwd, rev = pdata._build_identifier_maps(pdata.pep)
        assert fwd == {} and rev == {}

# ----------------------------------------------------------------------
# update_missing_genes coverage
def test_update_missing_genes_missing_column(pdata, capsys):
    """Covers branch where gene_col not found."""
    pdata.prot.var.rename(columns={"Genes": "WrongName"}, inplace=True)
    pdata.update_missing_genes(gene_col="Genes", verbose=True)
    out = capsys.readouterr().out
    assert "Column 'Genes' not found" in out

def test_update_missing_genes_no_missing(monkeypatch, pdata, capsys):
    """Covers branch when no missing entries."""
    monkeypatch.setattr("scpviz.utils.get_uniprot_fields", lambda *a, **k: pd.DataFrame())
    # fill all Genes → no missing mask
    pdata.prot.var["Genes"] = pdata.prot.var["Genes"].fillna("TESTGENE")
    pdata.update_missing_genes(verbose=True)
    out = capsys.readouterr().out
    assert "No missing gene names" in out

# ----------------------------------------------------------------------
# search_annotations coverage
def test_search_annotations_case_and_all_matches(pdata):
    """Covers return_all_matches=False branch and case=True logic."""
    df = pdata.search_annotations("some_term", on="protein", return_all_matches=False, case=True)
    assert isinstance(df, pd.DataFrame)
    assert "Matched" in df.columns

def test_validate_passes_on_valid_pdata(pdata, capsys):
    """Ensure validate() returns True and prints success for valid objects."""
    result = pdata.validate(verbose=True)
    captured = capsys.readouterr().out

    assert result is True
    assert "is valid" in captured

def test_validate_detects_obs_var_shape_mismatch(pdata, capsys):
    # Introduce duplicated var index to simulate structural inconsistency
    pdata.prot.var.index = pdata.prot.var.index.to_series().replace(pdata.prot.var.index[0], pdata.prot.var.index[1])
    
    result = pdata.validate(verbose=True)
    captured = capsys.readouterr().out

    assert result is False
    assert "prot.var has duplicated index" in captured

def test_validate_detects_obs_name_mismatch(pdata, capsys):
    pdata.pep.obs.index = [f"s{i}" for i in range(len(pdata.pep))]
    result = pdata.validate(verbose=True)
    captured = capsys.readouterr().out

    assert result is False
    assert "obs_names do not match" in captured

def test_validate_detects_rs_shape_mismatch(pdata, capsys):
    from scipy import sparse
    import numpy as np
    
    bad_rs = sparse.csr_matrix(np.zeros((pdata.prot.shape[1] - 1, pdata.pep.shape[1])))
    pdata._set_RS(bad_rs, validate=False)  # ⬅️ bypass shape check

    result = pdata.validate(verbose=True)
    captured = capsys.readouterr().out

    assert result is False
    assert "RS shape mismatch" in captured

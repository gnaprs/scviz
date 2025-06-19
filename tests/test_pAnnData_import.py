import pytest
from scviz import pAnnData
from pathlib import Path

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

def test_import_diann_old():
    # pre diann v1.8.1
    test_dir = Path(__file__).parent
    diann_file = str(test_dir / 'test_diann.tsv')

    obs_columns = ['name','amt','enzyme','date','MS','acquisition','method','gradient','replicate']
    pdata = pAnnData.import_data(source_type='diann', file=diann_file, obs_columns=obs_columns)
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
    pdata = pAnnData.import_data(source_type='diann', file=diann_file, obs_columns=obs_columns)
    assert pdata is not None
    assert pdata.prot is not None
    assert pdata.pep is not None
    assert pdata.rs is not None

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

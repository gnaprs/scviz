import pytest
from scviz import pAnnData
from pathlib import Path

def create_dummy_pAnnData():
    """Creates a dummy pAnnData object for testing."""
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot.txt')
    pep_file = str(test_dir / 'test_pd_pep.txt')

    obs_columns = ['sample', 'cellline', 'treatment']
    pdata = pAnnData.import_proteomeDiscoverer(prot_file = prot_file, pep_file = pep_file, obs_columns=obs_columns)
    
    return pdata

@pytest.fixture
def pdata():
    """Fixture to provide a dummy pAnnData object."""
    return create_dummy_pAnnData()
from scviz import utils as scutils
import pytest

from scviz import pAnnData

from pathlib import Path

def create_dummy_pAnnData():
    """Creates a dummy pAnnData object for testing."""
    # data = np.random.rand(100, 10)
    # prot_var = pd.DataFrame({
    #     'Protein FDR Confidence: Combined': ['High'] * 50 + ['Low'] * 50,
    #     'Description': ['p97 subunit'] * 100
    # })
    # pep_var = pd.DataFrame({
    #     'Score': np.random.rand(100),
    #     'Protein Confidence Level': [95] * 50 + [90] * 50
    # })

    # pdata = pAnnData(data, prot_var=prot_var, pep_var=pep_var)

    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'pd_prot_short.txt')
    pep_file = str(test_dir / 'pd_pep_short.txt')

    obs_columns = ['sample', 'cellline', 'treatment']
    pdata = pAnnData.import_proteomeDiscoverer(prot_file = prot_file, pep_file = pep_file, obs_columns=obs_columns)
    
    return pdata

@pytest.fixture
def pdata():
    """Fixture to provide a dummy pAnnData object."""
    return create_dummy_pAnnData()
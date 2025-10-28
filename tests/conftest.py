import pytest
from scviz.pAnnData import pAnnData
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("Agg")

def create_pd_pAnnData():
    """Creates a pd pAnnData object for testing."""
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot.txt')
    pep_file = str(test_dir / 'test_pd_pep.txt')

    obs_columns = ['sample', 'cellline', 'treatment']
    pdata = pAnnData.import_data(source_type="pd", prot_file=prot_file, pep_file=pep_file, obs_columns=obs_columns)

    return pdata

@pytest.fixture
def pdata():
    """Fixture to provide a pd pAnnData object."""
    return create_pd_pAnnData()

def create_pd_pAnnData_nopep():
    """Creates a pd pAnnData object for testing with no peptide."""
    test_dir = Path(__file__).parent
    prot_file = str(test_dir / 'test_pd_prot.txt')

    obs_columns = ['sample', 'cellline', 'treatment']
    pdata = pAnnData.import_data(source_type="pd", prot_file=prot_file, obs_columns=obs_columns)

    return pdata

@pytest.fixture
def pdata_nopep():
    """Fixture to provide a pd pAnnData object with no peptide."""
    return create_pd_pAnnData_nopep()

def create_diann_pAnnData():
    """Creates a DIA-NN-based pAnnData object with per-sample q-values."""
    test_dir = Path(__file__).parent
    diann_file = str(test_dir / "test_diann.parquet")

    obs_columns = [
        "name", "date", "MS", "acquisition", "FAIMS",
        "column", "gradient", "amt", "region", "replicate",
    ]
    pdata = pAnnData.import_data(
        source_type="diann",
        report_file=diann_file,
        obs_columns=obs_columns,
    )

    # Confirm q-value layers exist (normally DIA-NN always provides these)
    if "X_qval" not in pdata.prot.layers:
        n_obs, n_var = pdata.prot.shape
        pdata.prot.layers["X_qval"] = np.random.rand(n_obs, n_var)

    return pdata

@pytest.fixture
def pdata_diann():
    """DIA-NN–based fixture for tests requiring per-sample q-value data."""
    return create_diann_pAnnData()

# helper for plotting tests
def _is_axes_container(obj):
    """
    Return True if `obj` is a matplotlib Axes or a container (list/array)
    of Axes objects.
    """
    if isinstance(obj, plt.Axes):
        return True
    if isinstance(obj, (list, np.ndarray)):
        return all(isinstance(a, plt.Axes) for a in obj)
    return False

def _count_artists(ax):
    """Return number of drawable artists (patches, lines, collections)."""
    if isinstance(ax, (list, np.ndarray)):
        return sum(len(a.collections) + len(a.patches) + len(a.lines) for a in ax)
    return len(ax.collections) + len(ax.patches) + len(ax.lines)

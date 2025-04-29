import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scviz import pAnnData
from tests.conftest import pdata

@pytest.fixture
def pdata_preprocessing():
    """Fixture for controlled dummy data to test imputation."""
    X = np.array([
        [1,    np.nan, 10,   100],
        [2,    20,     np.nan, 200],
        [np.nan, 30,   30,   np.nan],
        [100,  np.nan, 1000, 500],
        [200,  400,     np.nan, np.nan],
        [np.nan, 600,  3000, 1500],
    ])

    obs = pd.DataFrame({
        "cellline": ["BE", "BE", "BE", "AS", "AS", "AS"],
        "treatment": ["kd", "kd", "kd", "sc", "sc", "sc"]
    }, index=[f"sample{i+1}" for i in range(6)])

    var = pd.DataFrame(index=[f"P{i+1}" for i in range(4)])
    ann = AnnData(X=X, obs=obs, var=var)
    return pAnnData.pAnnData(prot=ann)

def test_impute_mean_groupwise(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.impute(classes=["cellline", "treatment"], method="mean")
    imputed = pdata.prot.X

    # Check all NaNs have been filled
    assert not np.isnan(imputed).any(), "There should be no NaNs after mean group-wise imputation."
    assert np.isclose(imputed[0, 1], 25.0) # BE_kd1 was missing P2 → (20 + 30) / 2 = 25
    assert np.isclose(imputed[1, 2], 20.0) # BE_kd2 missing P3 → (10 + 30) / 2 = 20
    assert np.isclose(imputed[2, 0], 1.5) # BE_kd3 missing P1 → (1 + 2) / 2 = 1.5
    assert np.isclose(imputed[3, 1], 500.0) # AS_sc1 missing P2 → (400 + 600) / 2 = 500

def test_impute_median_groupwise(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.impute(classes=["cellline", "treatment"], method="median")
    imputed = pdata.prot.X

    # Ensure all imputable NaNs are filled
    assert not np.isnan(imputed).any(), "There should be no NaNs after median group-wise imputation."
    assert np.isclose(imputed[0, 1], 25.0)    # BE_kd1 was missing P2 → median of [20, 30] = 25
    assert np.isclose(imputed[1, 2], 20.0)    # BE_kd2 was missing P3 → median of [10, 30] = 20
    assert np.isclose(imputed[2, 0], 1.5)     # BE_kd3 was missing P1 → median of [1, 2] = 1.5
    assert np.isclose(imputed[3, 1], 500.0)   # AS_sc1 was missing P2 → median of [400, 600] = 500
    assert np.isclose(imputed[4, 2], 2000.0)  # AS_sc2 was missing P3 → median of [1000, 3000] = 2000
    assert np.isclose(imputed[4, 3], 1000.0)  # AS_sc2 was missing P4 → median of [500, 1500] = 1000

def test_impute_knn(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.impute(method="knn", n_neighbors=2)
    imputed = pdata.prot.X

    # Check shape and that no NaNs remain
    assert imputed.shape == pdata.prot.shape, "Shape mismatch after KNN imputation."
    assert not np.isnan(imputed).any(), "There should be no NaNs after KNN imputation."

    val = imputed[0, 1]  # P2 for BE_kd1, should be between 20 and 30
    assert 20 <= val <= 30, f"KNN-imputed value {val} out of expected range"

def test_impute_min_sparse(pdata_preprocessing):
    pdata = pdata_preprocessing
    # Convert to sparse before imputation
    from scipy import sparse
    
    pdata.prot.X = sparse.csr_matrix(pdata.prot.X)
    original = pdata_preprocessing.prot.X.toarray() if sparse.issparse(pdata_preprocessing.prot.X) else pdata_preprocessing.prot.X.copy()
    pdata.impute(method="min")
    imputed = pdata.prot.X

    # Check result is still sparse and no NaNs remain
    assert sparse.issparse(imputed), "Output should be sparse after min imputation."
    assert not np.isnan(imputed.toarray()).any(), "There should be no NaNs after min imputation."
    
    group = original[[0, 1, 2], :]
    expected = np.nanmin(group, axis=0)[1]  # min for P2 in BE_kd
    assert np.isclose(imputed.toarray()[0, 1], expected)

def test_impute_invalid_method(pdata_preprocessing):
    pdata = pdata_preprocessing
    with pytest.raises(ValueError, match="Unsupported method"):
        pdata.impute(method="invalid_method")

def test_impute_set_X_overwrites(pdata_preprocessing):
    pdata = pdata_preprocessing
    # Save original .X
    original = pdata.prot.X.copy()
    pdata.impute(method="mean", set_X=True)
    # Check if .X was overwritten
    assert not np.allclose(original, pdata.prot.X), "Expected .X to be updated after imputation."

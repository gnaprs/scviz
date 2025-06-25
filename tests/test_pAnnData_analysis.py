import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scviz import pAnnData
from tests.conftest import pdata

@pytest.fixture
def pdata_preprocessing():
    X = np.array([
        [1,    np.nan, 10,   100, 500, 2.0],
        [2,    20,     np.nan, 200, 500, 2.5],
        [np.nan, 30,   30,   np.nan, 500, 3.0],
        [100,  np.nan, 1000, 500, 500, 2.8],
        [200,  400,     np.nan, np.nan, 500, 2.2],
        [np.nan, 600,  3000, 1500, 500, 2.1],
    ])

    obs = pd.DataFrame({
        "cellline": ["BE", "BE", "BE", "AS", "AS", "AS"],
        "treatment": ["kd", "kd", "kd", "sc", "sc", "sc"]
    }, index=[f"sample{i+1}" for i in range(6)])

    var = pd.DataFrame({
        "Genes": ["GAPDH", "ACTB", "TUBB", "MYH9", "HSP90", "RPLP0"]
    }, index=[f"P{i+1}" for i in range(6)])

    ann = AnnData(X=X, obs=obs, var=var)
    return pAnnData(prot=ann)

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
    pdata.impute(method="mean")
    # Check if .X was overwritten
    assert not np.allclose(original, pdata.prot.X), "Expected .X to be updated after imputation."

def test_impute_median_groupwise_skips_allnan_feature(pdata_preprocessing):
    pdata = pdata_preprocessing
    # Set entire column (P6) to NaN in BE_kd group
    pdata.prot.X[0:3, 5] = np.nan

    pdata.impute(classes=["cellline", "treatment"], method="median")
    imputed = pdata.prot.X

    # Check that P6 in BE_kd group is still NaN
    assert np.isnan(imputed[0:3, 5]).all(), "All-NaN feature in group should remain NaN after median imputation."

    # Check that AS_sc group (samples 3–5) still imputed P6 correctly
    assert not np.isnan(imputed[3:, 5]).any(), "Non-empty group should have values imputed."

def test_impute_median_global_skips_allnan_feature(pdata_preprocessing):
    pdata = pdata_preprocessing
    # Set entire column (e.g., P6) to NaN globally
    pdata.prot.X[:, 5] = np.nan

    pdata.impute(method="median")  # global median imputation
    imputed = pdata.prot.X

    # Check that P6 is still all NaN
    assert np.isnan(imputed[:, 5]).all(), "All-NaN feature should remain NaN after global median imputation."

    # Check that other missing values were imputed
    assert not np.isnan(imputed[:, :5]).any(), "All imputable values should be filled."

def test_normalize_sum(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method="sum", set_X=True)
    norm = pdata.prot.X
    row_sums = np.nansum(norm, axis=1)
    assert np.allclose(row_sums, row_sums[0]), "All row sums should match after sum normalization"

def test_normalize_sum_use_nonmissing(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method="sum", set_X=False, use_nonmissing=True)
    norm = pdata.prot.layers["X_norm_sum"]
    assert norm.shape == pdata.prot.shape
    assert np.isclose(norm[0, 0], 1.00199, atol=1e-3)
    assert np.isclose(norm[0, 2], 10.0199, atol=1e-3)
    assert np.isclose(norm[0, 4], 500.996, atol=1e-3)

def test_reference_feature_gene_name(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method="reference_feature", reference_columns=["GAPDH", "ACTB"], reference_method="mean", set_X=False)
    norm = pdata.prot.layers["X_norm_reference_feature"]
    assert norm.shape == pdata.prot.shape
    assert np.isclose(norm[0, 0], 200.0)
    assert np.isclose(norm[0, 2], 2000.0)
    assert np.isclose(norm[0, 4], 100000.0)

def test_normalize_median_groupwise(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method="median", classes=["cellline", "treatment"], set_X=False)
    norm = pdata.prot.layers["X_norm_median"]

    # Check scaling applied correctly within BE_kd
    assert np.isclose(norm[0, 0], 3.0)
    assert np.isclose(norm[0, 2], 30.0)
    assert np.isclose(norm[0, 5], 6.0)

    # Check that AS_sc group is also scaled correctly
    assert np.isclose(norm[3, 2], 1200.0)
    assert np.isclose(norm[4, 1], 800.0)
    assert np.isclose(norm[5, 2], 3000.0)

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scpviz import pAnnData
import scipy.sparse
from copy import deepcopy

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

# test impute

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

def test_impute_min_groupwise(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.impute(classes=["cellline", "treatment"], method="min")
    imputed = pdata.prot.X

    assert not np.isnan(imputed).any(), "There should be no NaNs after median group-wise imputation."

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

def test_impute_knn_groupwise_raises(pdata_preprocessing):
    pdata = pdata_preprocessing
    with pytest.raises(ValueError, match="KNN imputation is not supported for group-wise"):
        pdata.impute(classes='cellline',method="knn", n_neighbors=2)

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_impute_raises_if_layer_not_found(pdata, on):
    with pytest.raises(ValueError, match="Layer 'X_invalid' not found"):
        pdata.impute(on=on, layer="X_invalid")

# test normalize()

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

def test_normalize_reference_feature_by_gene_name(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method="reference_feature", reference_columns=["GAPDH", "ACTB"], reference_method="mean", set_X=False)
    norm = pdata.prot.layers["X_norm_reference_feature"]
    assert norm.shape == pdata.prot.shape
    assert np.isclose(norm[0, 0], 200.0)
    assert np.isclose(norm[0, 2], 2000.0)
    assert np.isclose(norm[0, 4], 100000.0)

def test_normalize_reference_feature_by_index(pdata_preprocessing):
    pdata_preprocessing.normalize(
        method="reference_feature",
        reference_columns=["P1", "P2"],  # raw indices
        reference_method="mean",
        set_X=False
    )
    norm = pdata_preprocessing.prot.layers["X_norm_reference_feature"]
    assert np.isclose(norm[0, 0], 200.0)

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

def test_normalize_reference_feature_groupwise(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(
        method="reference_feature",
        reference_columns=["GAPDH", "ACTB"],
        reference_method="mean",
        classes=["cellline"],
        set_X=False
    )
    norm = pdata.prot.layers["X_norm_reference_feature"]
    expected = np.array([
        [2.0,      np.nan,   20.0,   200.0, 1000.0,   4.0  ],  # 2.0 × row 1
        [2.5,      25.0,     np.nan, 250.0,  625.0,   3.125],  # 1.25 × row 2
        [np.nan,   30.0,     30.0,   np.nan, 500.0,   3.0  ],  # 1.0 × row 3
        [200.0,    np.nan, 2000.0,  1000.0, 1000.0,   5.6  ],  # 2.0 × row 4
        [250.0,   500.0,     np.nan,  np.nan, 625.0,  2.75 ],  # 1.25 × row 5
        [np.nan,  600.0,   3000.0,  1500.0,  500.0,   2.1  ],  # 1.0 × row 6
    ])
    np.testing.assert_allclose(norm, expected, rtol=1e-4)

def test_normalize_warns_on_bad_rows(pdata_preprocessing, capsys):
    """Test that normalize() detects bad rows and exits early."""
    pdata = pdata_preprocessing

    # Force >50% NaN in one sample to trigger the warning
    pdata.prot.X[0, :] = np.nan  # first row = completely missing

    # Run normalization (without force=True) — should trigger early return
    pdata.normalize(method="sum")

    # Capture printed output
    captured = capsys.readouterr().out

    # Assert that the warning message was printed
    assert "have >50% missing values" in captured
    assert "Use `force=True` to proceed" in captured

    # Also assert that normalization did not proceed (layer not created)
    assert "X_norm_sum" not in pdata.prot.layers

def test_normalize_force_bad_rows(pdata_preprocessing, capsys):
    """Test that normalize(force=True) proceeds despite bad rows."""
    pdata = pdata_preprocessing
    pdata.prot.X[0, :] = np.nan

    pdata.normalize(method="sum", force=True)

    captured = capsys.readouterr().out
    assert "Proceeding with normalization despite bad rows" in captured
    assert "X_norm_sum" in pdata.prot.layers

@pytest.mark.xfail(reason="directlfq fails on some pandas versions (odule 'pandas.core.strings' has no attribute 'StringMethods')")
def test_normalize_directlfq(pdata):
    pdata.normalize(method='directlfq')
    assert 'X_norm_directlfq' in pdata.prot.layers

def test_normalize_robust_scale(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method='robust_scale')
    assert True

def test_normalize_quantile_transform(pdata_preprocessing):
    pdata = pdata_preprocessing
    pdata.normalize(method='quantile_transform')
    assert True

def test_normalize_invalid_method(pdata_preprocessing):
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        pdata_preprocessing.normalize(method="unknown_method")

# test de()

@pytest.mark.parametrize("fold_change_mode", ["mean", "pairwise_median"])
@pytest.mark.parametrize("test", ["ttest", "mannwhitneyu", "wilcoxon"])
def test_de_passes_on_valid_inputs(pdata, fold_change_mode, test):
    df = pdata.de(
        values=[
            {"cellline": "BE", "treatment": "kd"},
            {"cellline": "AS", "treatment": "sc"}
        ],
        method=test,
        fold_change_mode=fold_change_mode
    )
    assert isinstance(df, pd.DataFrame)
    assert "p_value" in df.columns
    assert "log2fc" in df.columns
    assert "[{'cellline': 'BE', 'treatment': 'kd'}]" in df.columns
    assert "[{'cellline': 'AS', 'treatment': 'sc'}]" in df.columns

def test_de_raises_on_invalid_foldchange(pdata):
    with pytest.raises(ValueError, match="Unsupported fold_change_mode"):
        pdata.de(
            values=[
                {"cellline": "BE"},
                {"cellline": "AS"}
            ],
            fold_change_mode="bogus"
        )

def test_de_raises_on_single_class(pdata):
    with pytest.raises(ValueError, match="provide two distinct groups"):
        pdata.de(values=[{"cellline": "BE"}, {"cellline": "BE"}])

def test_de_with_pep_pairwise_warns_if_no_pep(pdata_nopep):
    with pytest.raises(ValueError, match="Peptide-level data | required"):
        pdata_nopep.de(
            values=[{"cellline": "BE"}, {"cellline": "AS"}],
            fold_change_mode="pep_pairwise_median"
        )

def test_de_with_pep_pairwise_median(pdata):
    df = pdata.de(
            values=[
                {"cellline": "BE"},
                {"cellline": "AS"}
            ],
            fold_change_mode="pep_pairwise_median"        
    )

    assert isinstance(df, pd.DataFrame)
    assert "p_value" in df.columns
    assert "log2fc" in df.columns    

def test_de_ignores_inf_foldchange_in_annotations(pdata):
    pdata = deepcopy(pdata)

    X_orig = pdata.prot.X
    if scipy.sparse.issparse(X_orig):
        X_dense = X_orig.toarray()
    else:
        X_dense = X_orig.copy()

    # Get index of target protein (e.g., first column)
    prot_name = pdata.prot.var_names[0]
    prot_idx = pdata.prot.var_names.get_loc(prot_name)

    # Get all sample indices for each group
    group1_idx = np.where(pdata.prot.obs["cellline"] == "AS")[0]
    group2_idx = np.where(pdata.prot.obs["cellline"] == "BE")[0]

    # Inject values to force divide-by-zero log2FC
    X_dense[group1_idx, prot_idx] = 1e5  # large value
    X_dense[group2_idx, prot_idx] = 0    # zero

    # Set the modified matrix back to pdata.prot.X
    pdata.prot.X = scipy.sparse.csr_matrix(X_dense)

    # Run DE
    df = pdata.de(values=[{'cellline': 'AS'}, {'cellline':'BE'}], fold_change_mode="mean")

    print(df)

    assert prot_name in df.index, f"{prot_name} not found in DE result index"

    # Sanity check: log2FC should be +inf
    fc_val = df.loc[prot_name, "log2fc"]
    assert np.isnan(fc_val), f"Expected NaN log2fc for not comparable protein, got {fc_val}"
    assert df.loc[prot_name, "significance"] == "not comparable"

# test cv()

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_cv_computes_single_class(pdata, on):
    pdata.cv(classes="cellline", on=on)
    adata = pdata.prot if on == "protein" else pdata.pep
    var = adata.var

    for cls in ["AS", "BE"]:
        key = f"CV: {cls}"
        assert key in var.columns, f"Missing column {key}"
        assert len(var[key]) == adata.shape[1]
        # Allow NaNs but ensure majority of entries are valid
        valid_fraction = np.isfinite(var[key]).mean()
        assert valid_fraction > 0.8, f"Too many NaNs in {key} ({valid_fraction:.2%} valid)"

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_cv_computes_multi_class(pdata, on):
    pdata.cv(classes=["cellline", "treatment"], on=on)
    adata = pdata.prot if on == "protein" else pdata.pep
    var = adata.var

    expected_keys = [f"CV: {a}_{b}" for a in ["AS", "BE"] for b in ["sc", "kd"]]
    for key in expected_keys:
        assert key in var.columns, f"Missing column {key}"
        valid_fraction = np.isfinite(var[key]).mean()
        assert valid_fraction > 0.8, f"Too many NaNs in {key} ({valid_fraction:.2%} valid)"

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_cv_raises_if_layer_not_found(pdata, on):
    with pytest.raises(ValueError, match="Layer 'X_invalid' not found"):
        pdata.cv(classes="cellline", on=on, layer="X_invalid")

# test rank()

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_rank_computes_single_class(pdata, on):
    pdata.rank(classes="cellline", on=on)
    adata = pdata.prot if on == "protein" else pdata.pep
    var = adata.var

    for cls in ["AS", "BE"]:
        avg_key = f"Average: {cls}"
        std_key = f"Stdev: {cls}"
        rank_key = f"Rank: {cls}"

        # Columns should exist
        assert avg_key in var.columns
        assert std_key in var.columns
        assert rank_key in var.columns

        # Should be the same length as n_features
        assert len(var[avg_key]) == adata.shape[1]

        # Mean and Stdev: at least 90% finite
        assert np.isfinite(var[avg_key]).mean() > 0.9
        assert np.isfinite(var[std_key]).mean() > 0.9

        # Rank: Should be numeric or NaN, mostly finite
        assert var[rank_key].dtype.kind in ("f", "i")
        assert np.isfinite(var[rank_key]).mean() > 0.9

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_rank_computes_multi_class(pdata, on):
    pdata.rank(classes=["cellline", "treatment"], on=on)
    adata = pdata.prot if on == "protein" else pdata.pep
    var = adata.var

    for cls in ["AS_kd", "AS_sc", "BE_kd", "BE_sc"]:
        avg_key = f"Average: {cls}"
        std_key = f"Stdev: {cls}"
        rank_key = f"Rank: {cls}"

        assert avg_key in var.columns
        assert std_key in var.columns
        assert rank_key in var.columns
        assert np.isfinite(var[avg_key]).mean() > 0.9
        assert np.isfinite(var[std_key]).mean() > 0.9
        assert np.isfinite(var[rank_key]).mean() > 0.9

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_rank_raises_if_layer_not_found(pdata, on):
    with pytest.raises(ValueError, match="Layer 'X_invalid' not found"):
        pdata.rank(classes="cellline", on=on, layer="X_invalid")

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_rank_updates_history(pdata, on):
    prev_len = len(pdata._history)
    pdata.rank(classes="cellline", on=on)
    assert len(pdata._history) == prev_len + 1
    assert any("Ranked" in entry for entry in pdata._history[-1:])

def test_check_rankcol_raises_on_missing(pdata):
    with pytest.raises(ValueError, match="class_values must be None"):
        pdata._check_rankcol(on="protein", class_values=None)

# test neighbor()

@pytest.mark.parametrize("on", ["protein", "peptide"])
def test_neighbor_default_pca(pdata, on):
    if on == 'protein':
        adata_on='prot'
    else:
        adata_on='pep'
    # Delete existing PCA to force rerun
    getattr(pdata, adata_on).uns.pop("pca", None)
    getattr(pdata, adata_on).obsm.pop("X_pca", None)

    # Run neighbor without existing PCA
    pdata.neighbor(on=on)

    adata = getattr(pdata, adata_on)
    assert "neighbors" in adata.uns
    assert "distances" in adata.obsp
    assert "connectivities" in adata.obsp

    # Make sure distances and connectivities are square and match number of obs
    n = adata.shape[0]
    assert adata.obsp["distances"].shape == (n, n)
    assert adata.obsp["connectivities"].shape == (n, n)

def test_neighbor_custom_rep(pdata):
    # Run PCA to generate a valid rep
    pdata.pca(on="protein")
    adata = pdata.prot

    # Simulate an alternate embedding by copying X_pca to a new key
    adata.obsm["X_pca2"] = adata.obsm["X_pca"][:, :10]  # simulate reduced dims

    # Call neighbor with custom rep
    pdata.neighbor(on="protein", use_rep="X_pca2", n_neighbors=5)

    # Assertions
    assert "neighbors" in adata.uns
    assert "distances" in adata.obsp
    assert "connectivities" in adata.obsp
    assert adata.uns["neighbors"]["params"]["n_neighbors"] == 5
    assert adata.uns["neighbors"]["params"]["use_rep"] == "X_pca2"

def test_neighbor_invalid_rep_raises(pdata):
    with pytest.raises(ValueError, match="not found in obsm"):
        pdata.neighbor(on="protein", use_rep="X_fake")

def test_neighbor_with_layer_switch(pdata):
    from unittest.mock import patch
    pdata.prot.layers["dummy_layer"] = pdata.prot.X.copy()

    with patch.object(pdata, "set_X") as mocked:
        pdata.neighbor(on="protein", layer="dummy_layer")
        mocked.assert_called_once_with(layer="dummy_layer", on="protein")

    # Confirm results stored
    adata = pdata.prot
    assert "neighbors" in adata.uns
    assert "distances" in adata.obsp

# test leiden()

def test_leiden_default_layer(pdata):
    # Make sure no neighbors exist beforehand
    pdata.prot.uns.pop("neighbors", None)

    # Run Leiden clustering with default settings
    pdata.leiden(on="protein", layer="X", resolution=0.5)

    # Check results
    assert "leiden" in pdata.prot.obs.columns
    assert pdata.prot.obs["leiden"].notna().all()
    n_clusters = pdata.prot.obs["leiden"].nunique()
    assert n_clusters >= 1, "Leiden returned no labels"
    if n_clusters == 1:
        print("⚠️ Single cluster detected (likely due to small sample size or Scanpy≥1.10).")

from unittest.mock import patch

def test_leiden_custom_layer(pdata):
    # Add dummy layer and remove neighbors to trigger full path
    pdata.prot.layers["dummy"] = pdata.prot.X.copy()
    pdata.prot.uns.pop("neighbors", None)

    # Patch set_X to confirm it's called
    with patch.object(pdata, "set_X") as mocked:
        pdata.leiden(on="protein", layer="dummy", resolution=0.3)
        mocked.assert_called_with(layer="dummy", on="protein")  # Instead of .assert_called_once_with()
        assert mocked.call_count >= 1

    # Check Leiden results
    assert "leiden" in pdata.prot.obs.columns
    assert pdata.prot.obs["leiden"].notna().all()

def test_leiden_peptide_level(pdata):
    # Only run if .pep exists
    if pdata.pep is None:
        return

    pdata.pep.uns.pop("neighbors", None)
    pdata.leiden(on="peptide", layer="X", resolution=0.4)

    assert "leiden" in pdata.pep.obs.columns
    assert pdata.pep.obs["leiden"].notna().all()

# test umap()

def test_umap_runs_with_default_settings(pdata):
    # Precompute neighbors
    # Run UMAP
    pdata.umap(on="protein", layer="X")
    
    # Assert that UMAP coords are created
    assert "X_umap" in pdata.prot.obsm
    assert "umap" in pdata.prot.uns
    assert pdata.prot.obsm["X_umap"].shape[1] == 2

def test_umap_with_custom_umap_params_and_neighbors(pdata):
    # Remove neighbors to force recompute
    pdata.prot.uns.pop("neighbors", None)

    # Run UMAP with full parameter set
    pdata.umap(
        on="protein", layer="X",
        n_neighbors=10,
        n_pcs=20,
        min_dist=0.1,
        spread=1.0,
        metric="cosine",
        random_state=42
    )

    # Confirm UMAP output
    assert "X_umap" in pdata.prot.obsm
    assert "umap" in pdata.prot.uns
    coords = pdata.prot.obsm["X_umap"]
    assert coords.shape[0] == pdata.prot.n_obs
    assert coords.shape[1] == 2

from unittest.mock import patch

def test_umap_with_custom_layer_calls_set_X(pdata):
    # Add dummy layer
    pdata.prot.layers["dummy"] = pdata.prot.X.copy()
    pdata.prot.uns.pop("neighbors", None)

    # Patch set_X
    with patch.object(pdata, "set_X") as mocked:
        pdata.umap(on="protein", layer="dummy", n_neighbors=5)

        # Should call set_X before UMAP
        mocked.assert_called_with(layer="dummy", on="protein") 
        assert mocked.call_count >= 1

    # Confirm UMAP results still created
    assert "X_umap" in pdata.prot.obsm

def test_umap_on_peptide_data(pdata):
    # Confirm .pep exists
    assert pdata.pep is not None

    # Run UMAP on peptides
    pdata.umap(on="peptide", layer="X")

    # Confirm storage
    assert "X_umap" in pdata.pep.obsm
    assert "umap" in pdata.pep.uns

# test harmony
@pytest.mark.xfail(reason="Harmony fails on some Python/sklearn versions (n_clusters=0 bug)")
def test_harmony_runs_with_valid_key(pdata):
    # Sanity check
    assert pdata.prot.obs["cellline"].nunique() >= 2, "Cellline column must have ≥2 unique categories"

    if "X_pca" not in pdata.prot.obsm:
        pdata.pca(on="protein")

    # Run harmony
    pdata.harmony(key="cellline", on="protein")

    # Assertions
    assert "X_pca_harmony" in pdata.prot.obsm
    assert pdata.prot.obsm["X_pca_harmony"].shape[1] > 0
    assert "harmony" in "".join(pdata.history)  # optional

def test_harmony_invalid_key_raises(pdata):
    with pytest.raises(ValueError, match="Batch key 'invalid' not found"):
        pdata.harmony(key="invalid", on="protein")

@pytest.mark.xfail(reason="Harmony fails on some Python/sklearn versions (n_clusters=0 bug)")
def test_harmony_triggers_pca_if_missing(pdata):
    # Remove PCA
    pdata.prot.uns.pop("pca", None)
    pdata.prot.obsm.pop("X_pca", None)

    # Confirm PCA is not present
    assert "pca" not in pdata.prot.uns
    assert "X_pca" not in pdata.prot.obsm

    # Run harmony
    pdata.harmony(key="cellline", on="protein")

    # Confirm PCA was re-run
    assert "pca" in pdata.prot.uns
    assert "X_pca" in pdata.prot.obsm
    assert "X_pca_harmony" in pdata.prot.obsm

# test nanmissing values 

def test_nanmissingvalues_masks_exceeding_features(pdata):
    pdata = deepcopy(pdata)
    adata = pdata.prot

    # Convert to dense if needed
    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    # Inject missing values
    X[:, 0] = np.nan  # 100% missing → should be fully NaN
    X[:4, 1] = np.nan  # <50% missing → should be preserved
    X[:7, 2] = np.nan  # >50% missing → should be fully NaN

    adata.X = X
    pdata.prot = adata

    # Run filter
    pdata.nanmissingvalues(on="protein", limit=0.5)

    X_filtered = pdata.prot.X.toarray() if scipy.sparse.issparse(pdata.prot.X) else pdata.prot.X

    # Column 0 and 2 should be fully NaN
    assert np.all(np.isnan(X_filtered[:, 0]))
    assert np.all(np.isnan(X_filtered[:, 2]))

    # Column 1 should not be fully NaN
    assert not np.all(np.isnan(X_filtered[:, 1]))

def test_nanmissingvalues_supports_peptide(pdata):

    pdata = deepcopy(pdata)
    adata = pdata.pep

    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    # Inject 100% missing in col 0
    X[:, 0] = np.nan
    adata.X = X
    pdata.pep = adata

    # Run on peptide
    pdata.nanmissingvalues(on="peptide", limit=0.5)

    X_filtered = pdata.pep.X.toarray() if scipy.sparse.issparse(pdata.pep.X) else pdata.pep.X
    assert np.all(np.isnan(X_filtered[:, 0]))

def test_nanmissingvalues_limit_zero_masks_all_partial_missing(pdata):

    pdata = deepcopy(pdata)
    adata = pdata.prot

    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    # Inject some missing in column 3
    X[0, 3] = np.nan
    adata.X = X
    pdata.prot = adata

    # Run with 0.0 threshold
    pdata.nanmissingvalues(on="protein", limit=0.0)

    X_filtered = pdata.prot.X.toarray() if scipy.sparse.issparse(pdata.prot.X) else pdata.prot.X
    assert np.all(np.isnan(X_filtered[:, 3]))

# test clean_X

def test_clean_X_replaces_nans_in_dense_X(pdata):
    pdata = pdata.copy()
    pdata.prot.X[0, 0] = np.nan
    pdata.clean_X(on="prot", set_to=0, inplace=True)

    X = pdata.prot.X
    data = X.data if scipy.sparse.issparse(X) else X
    assert not np.isnan(data).any()

def test_clean_X_replaces_nans_in_sparse_X(pdata):
    pdata = pdata.copy()
    pdata.prot.X = scipy.sparse.csr_matrix(pdata.prot.X)
    pdata.prot.X.data[0] = np.nan
    pdata.clean_X(on="prot", set_to=7, inplace=True)

    assert not np.isnan(pdata.prot.X.data).any()
    assert 7 in pdata.prot.X.data

def test_clean_X_creates_backup_layer(pdata):
    pdata = pdata.copy()
    pdata.prot.X[0, 0] = np.nan
    pdata.clean_X(on="prot", set_to=0, backup_layer="X_testbackup")

    assert "X_testbackup" in pdata.prot.layers
    backup = pdata.prot.layers["X_testbackup"]
    data = backup.data if scipy.sparse.issparse(backup) else backup
    assert np.isnan(data).any()


def test_clean_X_returns_cleaned_matrix_when_inplace_false(pdata):
    pdata = pdata.copy()
    pdata.prot.X[0, 0] = np.nan
    cleaned = pdata.clean_X(on="prot", inplace=False, set_to=-1)

    assert scipy.sparse.issparse(cleaned) or isinstance(cleaned, np.ndarray)
    data = cleaned.data if scipy.sparse.issparse(cleaned) else cleaned
    assert not np.isnan(data).any()
    assert np.any(data == -1)

    # original should still contain NaN
    orig = pdata.prot.X
    orig_data = orig.data if scipy.sparse.issparse(orig) else orig
    assert np.isnan(orig_data).any()


def test_clean_X_to_sparse_returns_sparse_matrix(pdata):
    pdata = pdata.copy()
    pdata.prot.X[0, 0] = np.nan
    result = pdata.clean_X(on="prot", inplace=False, set_to=99, to_sparse=True)

    assert scipy.sparse.issparse(result)
    assert 99 in result.data

@pytest.mark.xfail(reason="to do")
def test_clean_X_works_on_peptide(pdata):
    pdata = pdata.copy()
    pdata.pep = pdata.prot.copy()
    pdata.pep.X[1, 1] = np.nan
    pdata.clean_X(on="peptide", set_to=42)

    X = pdata.pep.X
    data = X.data if scipy.sparse.issparse(X) else X

    val = pdata.pep.X[1, 1] if not scipy.sparse.issparse(X) else data[0]
    assert not np.isnan(val)
    assert np.any(data == 42)

def test_clean_X_layer_argument(pdata):
    pdata = pdata.copy()
    layer = pdata.prot.X.copy()
    layer[2, 2] = np.nan
    pdata.prot.layers["testlayer"] = layer
    pdata.clean_X(on="prot", layer="testlayer", set_to=777)

    layer_out = pdata.prot.layers["testlayer"]
    if scipy.sparse.issparse(layer_out):
        data = layer_out.data
    else:
        data = layer_out

    assert not np.isnan(data).any()
    assert np.any(data == 777)

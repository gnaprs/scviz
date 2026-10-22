import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from unittest.mock import patch, MagicMock
import anndata as ad

# -----------------------------
# Tests for set_X
# -----------------------------

def test_set_X_valid_layer(pdata_diann):
    from scipy.sparse import issparse
    # Add a fake layer and assign
    pdata_diann.prot.layers["test_layer"] = pdata_diann.prot.X.copy()
    pdata_diann.set_X(layer="test_layer", on="protein")

    x = pdata_diann.prot.X
    y = pdata_diann.prot.layers["test_layer"]

    # Convert to dense if sparse
    if issparse(x):
        x = x.toarray()
    if issparse(y):
        y = y.toarray()

    np.testing.assert_allclose(x, y)

def test_set_X_invalid_layer_raises(pdata_diann):
    with pytest.raises(ValueError, match="Layer .* not found"):
        pdata_diann.set_X(layer="nonexistent_layer", on="protein")

def test_set_X_updates_current_X_layer(pdata):
    """set_X() maintains current_X_layer in adata.uns (via normalize → set_X)."""
    pdata.normalize(method="median")
    assert pdata.prot.uns.get("current_X_layer") == "X_norm_median"

def test_set_X_false_does_not_update_current_X_layer(pdata):
    """set_X=False leaves current_X_layer unchanged."""
    initial = pdata.prot.uns.get("current_X_layer")
    pdata.normalize(method="median", set_X=False)
    assert pdata.prot.uns.get("current_X_layer") == initial

# -----------------------------
# Tests for get_abundance
# -----------------------------

def test_get_abundance_default(pdata_diann):
    df = pdata_diann.get_abundance(on="protein")
    assert isinstance(df, pd.DataFrame)
    assert "log2_abundance" in df.columns
    assert "x_label_name" in df.columns
    assert "class" in df.columns


def test_get_abundance_peptide_direct(pdata_diann):
    # Pick a few known peptides
    peptide_names = pdata_diann.pep.var_names[:3].tolist()
    df = pdata_diann.get_abundance(namelist=peptide_names, on="peptide", log=False)
    assert "abundance" in df.columns
    assert "log2_abundance" not in df.columns  # log=False
    assert all(df["accession"].isin(peptide_names))


def test_get_abundance_peptide_no_match_raises(pdata_diann):
    empty_pep = MagicMock()
    empty_pep.n_vars = 0

    empty_pdata = MagicMock()
    empty_pdata.pep = empty_pep

    with patch.object(pdata_diann, "filter_prot", return_value=empty_pdata):
        with pytest.raises(ValueError, match="No matching peptides found"):
            pdata_diann.get_abundance(namelist=["not_a_real_peptide_123"], on="peptide")

def test_get_abundance_all_matches_expands_isoforms(pdata):
    pdata = pdata.copy()
    dup_gene = "TEST_DUP_GENE_ABUND"
    idx = list(pdata.prot.var_names[:2])
    pdata.prot.var.loc[idx, "Genes"] = dup_gene

    df_one = pdata.get_abundance(namelist=[dup_gene], on="protein", log=False)
    df_all = pdata.get_abundance(
        namelist=[dup_gene], on="protein", log=False, all_matches=True
    )

    assert set(df_one["accession"]) == {idx[1]}
    assert set(df_all["accession"]) == set(idx)
    assert len(df_all) == len(df_one) * 2

def test_get_abundance_peptide_partial_match(pdata_diann):
    # Choose one valid peptide and one invalid one
    real_pep = pdata_diann.pep.var_names[0]
    fake_pep = "not_a_real_peptide_123"

    # Make a dummy return from filter_prot where .pep is None
    dummy_return = MagicMock()
    dummy_return.pep = None

    # Patch filter_prot so its internals are not executed (avoids IndexError)
    with patch.object(pdata_diann, "filter_prot", return_value=dummy_return):
        df = pdata_diann.get_abundance(
            namelist=[real_pep, fake_pep],
            on="peptide",
            log=False
        )

    # Basic assertions: returned dataframe contains the real peptide and not the fake one
    assert "abundance" in df.columns
    assert real_pep in df["accession"].unique()
    assert fake_pep not in df["accession"].unique()

# -----------------------------
# Tests for export
# -----------------------------

def test_export_creates_csv_files(pdata, tmp_path):
    output_prefix = tmp_path / "export_test"
    pdata.export(filename=str(output_prefix), verbose=False)

    expected_files = [
        f"{output_prefix}_summary.csv",
        f"{output_prefix}_protein.csv",
        f"{output_prefix}_peptide.csv",
    ]
    for file in expected_files:
        assert Path(file).exists()

    # Check at least one named protein layer file (skip layers[None] = .X on anndata >= 0.13)
    protein_layers = [k for k in pdata.prot.layers.keys() if isinstance(k, str)]
    if protein_layers:
        layer_file = f"{output_prefix}_protein_{protein_layers[0]}.csv"
        assert Path(layer_file).exists()

# -----------------------------
# Tests for export_layer
# -----------------------------

def _dense_layer(pdata, layer_name="X_qval", on="protein"):
    adata = pdata.prot if on == "protein" else pdata.pep
    layer = adata.layers[layer_name]
    return layer.toarray() if hasattr(layer, "toarray") else np.asarray(layer)

def test_export_layer_basic(pdata_diann, tmp_path):
    fname = tmp_path / "X_qval_export.csv"
    pdata_diann.export_layer("X_qval", filename=str(fname), on="protein")
    df = pd.read_csv(fname, index_col=0)
    assert df.shape == pdata_diann.prot.shape
    np.testing.assert_allclose(df.values, _dense_layer(pdata_diann))

def test_export_layer_single_label(pdata_diann, tmp_path):
    """One ``.obs`` column and one ``.var`` column as strings; flat CSV index/columns."""
    obs_col = pdata_diann.prot.obs.columns[0]
    var_col = "Genes" if "Genes" in pdata_diann.prot.var.columns else pdata_diann.prot.var.columns[0]
    fname = tmp_path / "X_qval_labeled.csv"
    pdata_diann.export_layer(
        "X_qval", filename=str(fname), on="protein", obs_names=obs_col, var_names=var_col
    )
    df = pd.read_csv(fname, index_col=0)
    assert df.shape == pdata_diann.prot.shape
    np.testing.assert_allclose(df.values, _dense_layer(pdata_diann))
    np.testing.assert_array_equal(
        df.index.astype(str).values,
        pdata_diann.prot.obs[obs_col].astype(str).values,
    )
    # Column labels duplicate ``Genes`` entries are mangled by CSV round-trip (``.1``, …).
    assert len(df.columns) == pdata_diann.prot.n_vars

def test_export_layer_multi_label(pdata_diann, tmp_path):
    """Lists of ``.obs`` / ``.var`` columns → MultiIndex rows/columns and two header rows."""
    obs_cols = ["name", "date"]
    var_cols = ["Genes", "Global_Q_value"]
    for c in obs_cols:
        assert c in pdata_diann.prot.obs.columns
    for c in var_cols:
        assert c in pdata_diann.prot.var.columns

    fname = tmp_path / "X_qval_multi.csv"
    pdata_diann.export_layer(
        "X_qval",
        filename=str(fname),
        on="protein",
        obs_names=obs_cols,
        var_names=var_cols,
    )
    df = pd.read_csv(fname, header=[0, 1], index_col=[0, 1])
    assert df.shape == pdata_diann.prot.shape
    assert df.index.nlevels == 2
    assert df.columns.nlevels == 2
    assert list(df.index.names) == obs_cols
    assert list(df.columns.names) == var_cols
    np.testing.assert_allclose(df.values, _dense_layer(pdata_diann))

def test_export_layer_transpose(pdata_diann, tmp_path):
    fname = tmp_path / "X_qval_T.csv"
    pdata_diann.export_layer("X_qval", filename=str(fname), on="protein", transpose=True)
    df = pd.read_csv(fname, index_col=0)
    assert df.shape == (pdata_diann.prot.shape[1], pdata_diann.prot.shape[0])
    np.testing.assert_allclose(df.values, _dense_layer(pdata_diann).T)

def test_export_layer_transpose_multi_labels(pdata_diann, tmp_path):
    """Transpose with MultiIndex on both axes: proteins as rows, samples as columns."""
    obs_cols = ["name", "date"]
    var_cols = ["Genes", "Global_Q_value"]
    for c in obs_cols:
        assert c in pdata_diann.prot.obs.columns
    for c in var_cols:
        assert c in pdata_diann.prot.var.columns

    fname = tmp_path / "X_qval_T_multi.csv"
    pdata_diann.export_layer(
        "X_qval",
        filename=str(fname),
        on="protein",
        obs_names=obs_cols,
        var_names=var_cols,
        transpose=True,
    )
    df = pd.read_csv(fname, header=[0, 1], index_col=[0, 1])
    layer = _dense_layer(pdata_diann)
    assert df.shape == (layer.shape[1], layer.shape[0])
    assert df.index.nlevels == 2
    assert df.columns.nlevels == 2
    assert list(df.index.names) == var_cols
    assert list(df.columns.names) == obs_cols
    np.testing.assert_allclose(df.values, layer.T)

# -----------------------------
# Tests for export_morpheus
# -----------------------------

def test_export_morpheus_outputs_files(pdata, tmp_path):
    fname_prefix = tmp_path / "morpheus_test"
    pdata.export_morpheus(filename=str(fname_prefix), on="protein")

    # All filenames
    f_matrix = f"{fname_prefix}_protein_matrix.csv"
    f_anno_obs = f"{fname_prefix}_protein_annotations.csv"

    assert Path(f_matrix).exists()
    assert Path(f_anno_obs).exists()

    df_matrix = pd.read_csv(f_matrix, index_col=0)
    df_obs = pd.read_csv(f_anno_obs, index_col=0)

    # df_obs should match obs (rows) or var (columns), depending on which file we parse
    assert df_matrix.shape == pdata.prot.X.toarray().shape

# -----------------------------
# Tests for set_rs
# -----------------------------

def test_set_rs_protein_by_peptide(pdata_diann):
    rs = np.ones((pdata_diann.prot.shape[1], pdata_diann.pep.shape[1]))  # valid shape
    pdata_diann._set_RS(rs, validate=True)
    assert pdata_diann._rs.shape == rs.shape

def test_set_rs_transpose_needed(pdata_diann):
    rs = np.ones((pdata_diann.pep.shape[1], pdata_diann.prot.shape[1]))  # transposed shape
    pdata_diann._set_RS(rs, validate=True)
    assert pdata_diann._rs.shape == (pdata_diann.prot.shape[1], pdata_diann.pep.shape[1])

def test_set_rs_invalid_shape(pdata_diann):
    rs = np.ones((10, 10))  # incorrect shape
    with pytest.raises(ValueError, match="RS shape .* does not match expected"):
        pdata_diann._set_RS(rs, validate=True)

def test_repr_skips_none_layer_key(pdata_diann, monkeypatch):
    """anndata >= 0.13 stores .X as layers[None]; __repr__ must skip non-string keys."""
    monkeypatch.setattr(pdata_diann.prot.layers, "keys", lambda: [None, "X_raw", "X_qval"])
    if pdata_diann.pep is not None:
        monkeypatch.setattr(pdata_diann.pep.layers, "keys", lambda: [None, "X_raw"])
    text = repr(pdata_diann)
    assert "X_raw" in text
    assert "X_qval" in text

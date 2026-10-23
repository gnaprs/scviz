"""Smoke tests on imported PD / DIA-NN fixtures (real file layout, not synthetic).

Logic and quantitative assertions belong in feature-specific test files with
synthetic data. These tests only verify that user-facing APIs run on real imports.
See `.cursor/rules/integration-testing.mdc`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scpviz import plotting as scplt
from scpviz import utils
from scpviz.pAnnData import pAnnData

TEST_DIR = Path(__file__).parent

PD_OBS_COLUMNS = ["Sample", "cellline", "treatment"]
DIANN_OBS_COLUMNS = [
    "name",
    "date",
    "MS",
    "acquisition",
    "FAIMS",
    "column",
    "gradient",
    "amt",
    "region",
    "replicate",
]

_MIXED_DE_KW = dict(
    observation_level="pseudobulk",
    min_detected_fraction=0.0,
    min_cells_detected=1,
)


@pytest.fixture(scope="module")
def pdata_pd_sample_column():
    """PD import with capital ``Sample`` column (matches notebook / Proteome Discoverer)."""
    return pAnnData.import_data(
        source_type="pd",
        prot_file=str(TEST_DIR / "test_pd_prot.txt"),
        pep_file=str(TEST_DIR / "test_pd_pep.txt"),
        obs_columns=PD_OBS_COLUMNS,
    )


def _accession_with_peptides(pdata):
    """Protein accession with the most RS-linked peptides."""
    counts = pdata.rs.getnnz(axis=1)
    idx = int(np.argmax(counts))
    return str(pdata.prot.var_names[idx]), int(counts[idx])


def _inject_column_abundance(pdata, acc: str, *, high_row: int = 0, high: float = 1e6, low: float = 1e2):
    """Set one feature column high in ``high_row`` and low elsewhere (in-place on copy)."""
    col = pdata.prot.var_names.get_loc(acc)
    n_obs = pdata.prot.n_obs
    values = np.where(np.arange(n_obs) == high_row, high, low)
    if sparse.issparse(pdata.prot.X):
        X = pdata.prot.X.toarray()
        X[:, col] = values
        pdata.prot.X = sparse.csr_matrix(X)
    else:
        pdata.prot.X[:, col] = values


def _subset_X_matrix(adata, X):
    if sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(X)
    else:
        adata.X = X


# ---------------------------------------------------------------------------
# mixed_de (existing coverage — no new tests)
# ---------------------------------------------------------------------------


def test_pd_de_cellline_values(pdata_pd_sample_column):
    """``de()`` on PD import with dict ``values`` sugar."""
    df = pdata_pd_sample_column.de(
        values=[{"cellline": "BE"}, {"cellline": "AS"}],
        fold_change_mode="mean",
    )
    assert "log2fc" in df.columns
    assert "p_value" in df.columns
    assert np.isfinite(df["p_value"]).any()


def test_pd_de_correct_fdr(pdata_pd_sample_column):
    """``de(correct_fdr=True)`` on PD with ``Sample`` metadata."""
    df = pdata_pd_sample_column.de(
        values=[{"cellline": "BE"}, {"cellline": "AS"}],
        correct_fdr=True,
        fold_change_mode="mean",
    )
    assert "adj_p_value" in df.columns
    assert np.isfinite(df["adj_p_value"]).any()


def test_pd_mixed_de_cellline_explicit_contrast(pdata_pd_sample_column):
    """``mixed_de()`` simple path on PD (``Sample`` column + summary merge)."""
    df = pdata_pd_sample_column.mixed_de(
        group_col="cellline",
        contrast=("BE", "AS"),
        donor_col="treatment",
        **_MIXED_DE_KW,
        store=False,
    )
    assert np.isfinite(df["p_value"]).any()


def test_pd_mixed_de_values_sugar_stores_stats_key(pdata_pd_sample_column):
    """``mixed_de()`` values sugar stores a retrievable stats key on PD data."""
    pdata_pd_sample_column.mixed_de(
        values=[{"cellline": "BE"}, {"cellline": "AS"}],
        donor_col="treatment",
        store=True,
        **_MIXED_DE_KW,
    )
    keys = [k for k in pdata_pd_sample_column.stats if k.startswith("mixed:")]
    assert any("BE vs AS" in k or "AS vs BE" in k for k in keys)
    key = next(k for k in keys if "donor=treatment" in k)
    assert "log2fc" in pdata_pd_sample_column.stats[key].columns


def test_pd_mixed_de_plot_volcano_stats_key(pdata_pd_sample_column):
    """``plot_volcano(stats_key=...)`` on PD ``mixed_de`` results."""
    import matplotlib.pyplot as plt

    pdata_pd_sample_column.mixed_de(
        group_col="cellline",
        contrast=("BE", "AS"),
        donor_col="treatment",
        store=True,
        **_MIXED_DE_KW,
    )
    key = next(
        k for k in pdata_pd_sample_column.stats if k.startswith("mixed:") and "donor=treatment" in k
    )
    fig, ax = plt.subplots(figsize=(3, 3))
    scplt.plot_volcano(ax, pdata_pd_sample_column, stats_key=key, correct_fdr=True)
    plt.close(fig)


def test_pd_log_transform_provenance(pdata_pd_sample_column):
    """``log_transform()`` registers layer provenance on PD import."""
    pdata = pdata_pd_sample_column.copy()
    pdata.log_transform(set_X=False)
    assert "X_log2" in pdata.prot.uns.get("layer_provenance", {})
    assert pdata.prot.uns["layer_provenance"]["X_log2"]["op"] == "log_transform"


def test_pd_pairwise_correlation(pdata_pd_sample_column):
    """``pairwise_correlation()`` on PD with ``Sample`` column."""
    pdata = pdata_pd_sample_column.copy()
    pdata.pairwise_correlation(classes="cellline")
    cache = pdata.prot.uns.get("pairwise_corr", {})
    assert "group_matrix" in cache
    mat = cache["group_matrix"]
    assert mat.shape[0] == mat.shape[1] >= 2


def test_pd_pca(pdata_pd_sample_column):
    """``pca()`` on PD import."""
    pdata = pdata_pd_sample_column.copy()
    pdata.pca(on="protein")
    assert "X_pca" in pdata.prot.obsm
    assert pdata.prot.obsm["X_pca"].shape[0] == pdata.prot.n_obs


def test_pd_cv(pdata_pd_sample_column):
    """``cv()`` on PD import."""
    pdata = pdata_pd_sample_column.copy()
    pdata.cv(classes="cellline", on="protein")
    assert "CV: AS" in pdata.prot.var.columns
    assert "CV: BE" in pdata.prot.var.columns


def test_pd_filter_sample_poi(pdata_pd_sample_column):
    """``filter_sample(poi=...)`` on PD import."""
    pdata = pdata_pd_sample_column.copy()
    acc = str(pdata.prot.var_names[0])
    _inject_column_abundance(pdata, acc)
    out = pdata.filter_sample(poi=acc, min_abundance=1e4, return_copy=True)
    assert 0 < out.prot.n_obs < pdata.prot.n_obs


def test_pd_resolve_peptides(pdata_pd_sample_column):
    """``resolve_peptides()`` expands accession to peptide IDs on PD."""
    acc, n_peps = _accession_with_peptides(pdata_pd_sample_column)
    pep_ids = utils.resolve_peptides(pdata_pd_sample_column, [acc], quiet=True)
    assert pep_ids is not None
    assert len(pep_ids) == n_peps


def test_pd_get_peptides_for_accessions(pdata_pd_sample_column):
    """``get_peptides_for_accessions()`` on PD import."""
    acc, n_peps = _accession_with_peptides(pdata_pd_sample_column)
    df = utils.get_peptides_for_accessions(pdata_pd_sample_column, [acc])
    assert list(df.columns) == ["accession", "peptide_id", "sequence"]
    assert len(df) == n_peps
    assert (df["accession"] == acc).all()


def test_pd_get_accessions_for_peptides_roundtrip(pdata_pd_sample_column):
    """Peptide ID roundtrip via RS on PD import."""
    acc, _ = _accession_with_peptides(pdata_pd_sample_column)
    pep_df = utils.get_peptides_for_accessions(pdata_pd_sample_column, [acc])
    peptide_id = pep_df.iloc[0]["peptide_id"]
    df = utils.get_accessions_for_peptides(pdata_pd_sample_column, [peptide_id])
    assert acc in df["accession"].values


def test_pd_get_peptide_properties(pdata_pd_sample_column):
    """``get_peptide_properties()`` on PD import."""
    acc, n_peps = _accession_with_peptides(pdata_pd_sample_column)
    df = utils.get_peptide_properties(pdata_pd_sample_column, accessions=[acc], return_copy=True)
    assert len(df) == n_peps
    assert df["gravy"].notna().any()
    assert "[" not in df["peptide_sequence"].iloc[0]


def test_pd_resolve_peptide_sequence(pdata_pd_sample_column):
    """``resolve_peptide_sequence()`` parses PD annotated peptide IDs."""
    pep_id = pdata_pd_sample_column.pep.var_names[0]
    seq = utils.resolve_peptide_sequence(pdata_pd_sample_column, pep_id)
    assert seq is not None
    assert "[" not in seq


def test_pd_update_summary_zero_as_missing(pdata_pd_sample_column):
    """Zeros in ``X`` count as missing in summary metrics on PD import."""
    pdata = pdata_pd_sample_column.copy()
    pdata.prot = pdata.prot[:4, :5].copy()
    _subset_X_matrix(pdata.prot, np.array([[1e5, 0.0, np.nan, 1.0, 0.0]] * 4))
    pdata.update_summary(recompute=True, verbose=False)
    assert pdata.prot.obs["protein_count"].between(1, 5).all()
    assert pdata.prot.obs["protein_quant"].between(0, 1).all()


# ---------------------------------------------------------------------------
# DIA-NN smoke tests
# ---------------------------------------------------------------------------


def test_diann_de_gradient_legacy_values(pdata_diann):
    """``de()`` on DIA-NN import (legacy class + values)."""
    df = pdata_diann.de(
        class_type="gradient",
        values=["30min", "60min"],
        fold_change_mode="mean",
    )
    assert np.isfinite(df["p_value"]).any()


def test_diann_de_values_correct_fdr(pdata_diann):
    """``de(values=..., correct_fdr=True)`` on DIA-NN import."""
    df = pdata_diann.de(
        values=[{"gradient": "30min"}, {"gradient": "60min"}],
        correct_fdr=True,
        fold_change_mode="mean",
    )
    assert "adj_p_value" in df.columns
    assert np.isfinite(df["adj_p_value"]).any()


def test_diann_mixed_de_gradient_replicate(pdata_diann):
    """``mixed_de()`` on DIA-NN with replicate blocking."""
    df = pdata_diann.mixed_de(
        group_col="gradient",
        contrast=("30min", "60min"),
        donor_col="replicate",
        **_MIXED_DE_KW,
        store=False,
    )
    assert "contrast" in df.columns
    assert df.attrs.get("mixed_de", {}).get("n_donors_paired", 0) >= 1
    assert np.isfinite(df["p_value"]).any()


def test_diann_mixed_de_values_sugar(pdata_diann):
    """``mixed_de()`` values sugar on DIA-NN import."""
    df = pdata_diann.mixed_de(
        values=[{"gradient": "30min"}, {"gradient": "60min"}],
        donor_col="replicate",
        **_MIXED_DE_KW,
        store=False,
    )
    assert df["contrast"].iloc[0] == "30min vs 60min"


def test_diann_mixed_de_resolve_obs_meta_matches_obs(pdata_diann):
    """Summary merge preserves DIA-NN obs metadata (no Sample column path)."""
    from scpviz.utils import mixed_de as mixed_de_utils

    meta = mixed_de_utils.resolve_obs_meta(
        pdata_diann.prot,
        pdata_diann.summary,
        ["gradient", "replicate", "region"],
    )
    pd.testing.assert_series_equal(
        meta["gradient"],
        pdata_diann.prot.obs["gradient"],
        check_names=False,
    )


def test_diann_log_transform_provenance(pdata_diann):
    """``log_transform()`` registers layer provenance on DIA-NN import."""
    pdata = pdata_diann.copy()
    pdata.log_transform(set_X=False)
    assert "X_log2" in pdata.prot.uns.get("layer_provenance", {})
    assert pdata.prot.uns["layer_provenance"]["X_log2"]["op"] == "log_transform"


def test_diann_pairwise_correlation(pdata_diann):
    """``pairwise_correlation()`` on DIA-NN import."""
    pdata = pdata_diann.copy()
    pdata.pairwise_correlation(classes="gradient")
    cache = pdata.prot.uns.get("pairwise_corr", {})
    assert "group_matrix" in cache
    assert cache["group_matrix"].shape[0] >= 2


def test_diann_pca(pdata_diann):
    """``pca()`` on DIA-NN import."""
    pdata = pdata_diann.copy()
    pdata.pca(on="protein")
    assert "X_pca" in pdata.prot.obsm
    assert pdata.prot.obsm["X_pca"].shape[0] == pdata.prot.n_obs


def test_diann_cv(pdata_diann):
    """``cv()`` on DIA-NN import."""
    pdata = pdata_diann.copy()
    pdata.cv(classes="gradient", on="protein")
    assert any(c.startswith("CV: ") for c in pdata.prot.var.columns)


def test_diann_filter_sample_poi(pdata_diann):
    """``filter_sample(poi=...)`` on DIA-NN import."""
    pdata = pdata_diann.copy()
    acc = str(pdata.prot.var_names[0])
    _inject_column_abundance(pdata, acc)
    out = pdata.filter_sample(poi=acc, min_abundance=1e4, return_copy=True)
    assert 0 < out.prot.n_obs < pdata.prot.n_obs


def test_diann_resolve_peptides(pdata_diann):
    """``resolve_peptides()`` on DIA-NN import."""
    acc = str(pdata_diann.prot.var_names[0])
    pep_df = utils.get_peptides_for_accessions(pdata_diann, [acc])
    if pep_df.empty:
        pytest.skip("No peptides linked to first protein in DIA-NN fixture")
    pep_ids = utils.resolve_peptides(pdata_diann, [acc], quiet=True)
    assert pep_ids is not None
    assert len(pep_ids) >= 1


def test_diann_get_peptides_for_accessions(pdata_diann):
    """``get_peptides_for_accessions()`` on DIA-NN import."""
    acc = str(pdata_diann.prot.var_names[0])
    df_index = utils.get_peptides_for_accessions(pdata_diann, [acc])
    if df_index.empty:
        pytest.skip("No peptides linked to first protein in DIA-NN fixture")
    assert (df_index["sequence"] == df_index["peptide_id"]).all()

    df_stripped = utils.get_peptides_for_accessions(
        pdata_diann, [acc], sequence_from="Stripped.Sequence"
    )
    assert not df_stripped.empty
    assert (df_stripped["sequence"] != df_stripped["peptide_id"]).any()


def test_diann_get_accessions_for_peptides_by_sequence(pdata_diann):
    """``get_accessions_for_peptides(sequence_from=...)`` on DIA-NN import."""
    stripped = str(pdata_diann.pep.var["Stripped.Sequence"].iloc[0])
    df = utils.get_accessions_for_peptides(
        pdata_diann, [stripped], sequence_from="Stripped.Sequence"
    )
    assert not df.empty
    assert (df["sequence"] == stripped).all()


def test_diann_get_peptide_properties(pdata_diann):
    """``get_peptide_properties()`` on DIA-NN import."""
    acc = str(pdata_diann.prot.var_names[0])
    df = utils.get_peptide_properties(pdata_diann, accessions=[acc], return_copy=True)
    if df.empty:
        pytest.skip("No peptides linked to first protein in DIA-NN fixture")
    assert df["peptide_sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+").all()
    row = df.iloc[0]
    stripped = str(pdata_diann.pep.var.loc[row["peptide_id"], "Stripped.Sequence"])
    assert row["peptide_sequence"] == stripped.upper()


def test_diann_resolve_peptide_sequence(pdata_diann):
    """``resolve_peptide_sequence()`` uses Stripped.Sequence on DIA-NN."""
    pep_id = pdata_diann.pep.var_names[0]
    stripped = str(pdata_diann.pep.var.loc[pep_id, "Stripped.Sequence"])
    seq = utils.resolve_peptide_sequence(pdata_diann, pep_id)
    assert seq == stripped.upper()
    assert pep_id != stripped


def test_diann_update_summary_zero_as_missing(pdata_diann):
    """Zeros in ``X`` count as missing in summary metrics on DIA-NN import."""
    pdata = pdata_diann.copy()
    pdata.prot = pdata.prot[:4, :5].copy()
    _subset_X_matrix(pdata.prot, np.array([[1e5, 0.0, np.nan, 1.0, 0.0]] * 4))
    pdata.update_summary(recompute=True, verbose=False)
    assert pdata.prot.obs["protein_count"].between(1, 5).all()
    assert pdata.prot.obs["protein_quant"].between(0, 1).all()


# ---------------------------------------------------------------------------
# Heatmap smoke tests (PD + DIA-NN)
# ---------------------------------------------------------------------------

def _heatmap_gene_groups(pdata, n=4):
    genes = pdata.prot.var["Genes"].dropna().astype(str).unique()[:n].tolist()
    return {
        "GroupA": genes[: max(1, n // 2)],
        "GroupB": genes[max(1, n // 2) :],
    }, genes


def test_pd_plot_grouped_heatmap(pdata_pd_sample_column):
    import matplotlib.pyplot as plt

    pdata = pdata_pd_sample_column
    groups, _ = _heatmap_gene_groups(pdata)
    classes = [c for c in ("treatment", "cellline") if c in pdata.prot.obs.columns]
    fig = scplt.plot_grouped_heatmap(
        pdata, protein_groups=groups, classes=classes, figsize=(7, 5)
    )
    assert fig is not None
    plt.close(fig)


def test_pd_plot_clustered_heatmap_proteins(pdata_pd_sample_column):
    import matplotlib.pyplot as plt

    pdata = pdata_pd_sample_column
    groups, genes = _heatmap_gene_groups(pdata, n=5)
    classes = [c for c in ("treatment", "cellline") if c in pdata.prot.obs.columns]
    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=classes,
        proteins=genes,
        protein_groups=groups,
        figsize=(7, 5),
    )
    assert fig is not None
    plt.close(fig)


def test_pd_plot_clustered_heatmap_stats_key(pdata_pd_sample_column):
    import matplotlib.pyplot as plt

    pdata = pdata_pd_sample_column
    accs = list(pdata.prot.var_names[:5].astype(str))
    de = pd.DataFrame(
        {
            "log2fc": [1.0, -1.0, 0.2, 1.5, -0.1],
            "p_value": [1e-4, 1e-3, 0.4, 1e-5, 0.8],
            "significance": [
                "upregulated",
                "downregulated",
                "not significant",
                "upregulated",
                "not significant",
            ],
        },
        index=accs,
    )
    key = "smoke_pd_heatmap_de"
    pdata.stats[key] = de
    classes = [c for c in ("treatment", "cellline") if c in pdata.prot.obs.columns]
    fig = scplt.plot_clustered_heatmap(
        pdata, classes=classes, stats_key=key, figsize=(7, 5)
    )
    assert fig is not None
    plt.close(fig)


def test_diann_plot_grouped_heatmap(pdata_diann):
    import matplotlib.pyplot as plt

    pdata = pdata_diann
    groups, _ = _heatmap_gene_groups(pdata)
    # DIA-NN fixture obs columns
    classes = [c for c in ("gradient", "region", "replicate") if c in pdata.prot.obs.columns]
    if not classes:
        classes = [pdata.prot.obs.columns[0]]
    fig = scplt.plot_grouped_heatmap(
        pdata, protein_groups=groups, classes=classes[:2], figsize=(7, 5)
    )
    assert fig is not None
    plt.close(fig)


def test_diann_plot_clustered_heatmap_proteins(pdata_diann):
    import matplotlib.pyplot as plt

    pdata = pdata_diann
    groups, genes = _heatmap_gene_groups(pdata, n=5)
    classes = [c for c in ("gradient", "region") if c in pdata.prot.obs.columns]
    if not classes:
        classes = [pdata.prot.obs.columns[0]]
    fig = scplt.plot_clustered_heatmap(
        pdata,
        classes=classes,
        proteins=genes,
        protein_groups=groups,
        show_unassigned=True,
        figsize=(7, 5),
    )
    assert fig is not None
    plt.close(fig)


def test_diann_plot_clustered_heatmap_mixin(pdata_diann):
    import matplotlib.pyplot as plt

    pdata = pdata_diann
    _, genes = _heatmap_gene_groups(pdata, n=4)
    classes = [c for c in ("gradient",) if c in pdata.prot.obs.columns] or [
        pdata.prot.obs.columns[0]
    ]
    fig = pdata.plot_clustered_heatmap(
        classes=classes, proteins=genes, metric="euclidean", figsize=(7, 5)
    )
    assert fig is not None
    plt.close(fig)


def test_pd_plot_clustermap(pdata_pd_sample_column):
    import matplotlib.pyplot as plt

    pdata = pdata_pd_sample_column
    classes = [c for c in ("treatment", "cellline") if c in pdata.prot.obs.columns]
    namelist = list(pdata.prot.var_names[:30].astype(str))
    fig = scplt.plot_clustermap(
        pdata, classes=classes, namelist=namelist, figsize=(7, 6)
    )
    assert fig is not None
    assert "prot_X_clustermap" in pdata.stats
    plt.close(fig)


def test_pd_plot_clustermap_mixin(pdata_pd_sample_column):
    import matplotlib.pyplot as plt

    pdata = pdata_pd_sample_column
    classes = [c for c in ("cellline",) if c in pdata.prot.obs.columns] or [
        pdata.prot.obs.columns[0]
    ]
    namelist = list(pdata.prot.var_names[:20].astype(str))
    fig = pdata.plot_clustermap(
        classes=classes, namelist=namelist, show_row_labels=True, figsize=(6, 5)
    )
    assert fig is not None
    plt.close(fig)


def test_diann_plot_clustermap(pdata_diann):
    import matplotlib.pyplot as plt

    pdata = pdata_diann
    classes = [c for c in ("gradient", "region") if c in pdata.prot.obs.columns]
    if not classes:
        classes = [pdata.prot.obs.columns[0]]
    namelist = list(pdata.prot.var_names[:30].astype(str))
    fig = scplt.plot_clustermap(
        pdata, classes=classes[:2], namelist=namelist, figsize=(7, 6)
    )
    assert fig is not None
    plt.close(fig)


def test_diann_plot_clustermap_separate_legend(pdata_diann):
    import matplotlib.pyplot as plt

    pdata = pdata_diann
    classes = [c for c in ("gradient",) if c in pdata.prot.obs.columns] or [
        pdata.prot.obs.columns[0]
    ]
    namelist = list(pdata.prot.var_names[:15].astype(str))
    out = scplt.plot_clustermap(
        pdata,
        classes=classes,
        namelist=namelist,
        separate_legend=True,
        figsize=(5, 4),
    )
    assert isinstance(out, tuple) and len(out) == 2
    plt.close(out[0])
    plt.close(out[1])


# ---------------------------------------------------------------------------
# STRING mapping cache
# ---------------------------------------------------------------------------


def test_pd_update_string_mapping_add(pdata_pd_sample_column):
    """PD: write a STRING ID onto `.prot.var`."""
    pdata = pdata_pd_sample_column.copy()
    acc = str(pdata.prot.var_names[0])
    pdata.update_string_mapping({acc: "9606.ENSP00000000001"}, verbose=False)
    assert pdata.prot.var.at[acc, "STRING"] == "9606.ENSP00000000001"
    assert pdata.metadata.get("string_map_history")


def test_pd_update_string_mapping_skip_existing(pdata_pd_sample_column):
    """PD: overwrite=False leaves an existing STRING ID unchanged."""
    pdata = pdata_pd_sample_column.copy()
    acc = str(pdata.prot.var_names[0])
    pdata.update_string_mapping({acc: "9606.OLD"}, verbose=False)
    pdata.update_string_mapping({acc: "9606.NEW"}, overwrite=False, verbose=False)
    assert pdata.prot.var.at[acc, "STRING"] == "9606.OLD"


def test_diann_update_string_mapping_with_taxon(pdata_diann):
    """DIA-NN: write STRING ID and taxon together."""
    pdata = pdata_diann.copy()
    acc = str(pdata.prot.var_names[0])
    pdata.update_string_mapping({acc: "9606.ENSP1"}, taxon=9606, verbose=False)
    assert pdata.prot.var.at[acc, "STRING"] == "9606.ENSP1"
    assert str(pdata.prot.var.at[acc, "ncbi_taxon_id"]) == "9606"


def test_diann_update_string_mapping_overwrite(pdata_diann):
    """DIA-NN: overwrite=True replaces a cached STRING ID."""
    pdata = pdata_diann.copy()
    acc = str(pdata.prot.var_names[0])
    pdata.update_string_mapping({acc: "9606.OLD"}, verbose=False)
    pdata.update_string_mapping({acc: "9606.NEW"}, overwrite=True, verbose=False)
    assert pdata.prot.var.at[acc, "STRING"] == "9606.NEW"

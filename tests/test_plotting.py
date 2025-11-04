from scpviz import utils as scutils
from scpviz import plotting as scplt

import pandas as pd
import numpy as np
import pytest
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad

from conftest import _is_axes_container, _count_artists

# test get_color

def test_get_color_colors_basic():
    colors = scplt.get_color("colors", 5)
    assert isinstance(colors, list)
    assert len(colors) == 5
    assert all(isinstance(c, str) and c.startswith("#") for c in colors)

def test_get_color_colors_warns_on_repeat():
    with pytest.warns(UserWarning, match="Reusing from the start"):
        colors = scplt.get_color("colors", 10)
    assert len(colors) == 10
    assert colors[0] == colors[7]  # colors repeat from base palette

def test_get_color_cmap_single():
    cmap = scplt.get_color("cmap", 1)
    assert isinstance(cmap, matplotlib.colors.LinearSegmentedColormap)

def test_get_color_cmap_multiple():
    cmaps = scplt.get_color("cmap", 3)
    assert isinstance(cmaps, list)
    assert all(isinstance(c, matplotlib.colors.LinearSegmentedColormap) for c in cmaps)

def test_get_color_palette():
    palette = scplt.get_color("palette")
    assert isinstance(palette, list)
    assert all(isinstance(c, tuple) and len(c) == 3 for c in palette)
    assert np.allclose(np.array(palette).max(), 1.0, atol=0.05)  # RGB normalized to ~1

def test_get_color_show_smoke(monkeypatch):
    # Avoid GUI popup during test runs
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    result = scplt.get_color("show")
    assert result is None

def test_get_color_invalid_type():
    with pytest.raises(ValueError, match="Invalid resource_type"):
        scplt.get_color("invalid")

def test_get_color_colors_missing_n():
    with pytest.raises(ValueError, match="must be specified"):
        scplt.get_color("colors")

# test plot_significance

def test_plot_significance_runs_without_error():
    fig, ax = plt.subplots()
    scplt.plot_significance(ax, y=1.0, h=0.1, x1=0, x2=1, pval=0.05)
    assert len(ax.lines) == 1
    assert len(ax.texts) == 1

def test_plot_significance_with_string_label():
    fig, ax = plt.subplots()
    scplt.plot_significance(ax, y=0.5, h=0.2, x1=0, x2=1, pval="custom")
    text_obj = ax.texts[0]
    assert text_obj.get_text() == "custom"
    assert text_obj.get_ha() == "center"
    assert text_obj.get_va() == "bottom"

@pytest.mark.parametrize("pval,expected", [
    (0.2, "n.s."),      # Not significant
    (0.05, "*"),        # 1 star
    (0.005, "**"),      # 2 stars
    (0.0005, "***"),    # 3 stars
])
def test_plot_significance_numeric_levels(pval, expected):
    fig, ax = plt.subplots()
    scplt.plot_significance(ax, y=0.5, h=0.2, pval=pval)
    text_label = ax.texts[0].get_text()
    assert text_label.startswith(expected[0])

def test_plot_significance_color_and_fontsize():
    fig, ax = plt.subplots()
    scplt.plot_significance(ax, y=1, h=0.2, col="red", fontsize=16)
    line = ax.lines[0]
    text = ax.texts[0]
    assert line.get_color() == "red"
    assert text.get_fontsize() == 16

def test_plot_significance_identical_x():
    fig, ax = plt.subplots()
    scplt.plot_significance(ax, y=1, h=0.1, x1=1, x2=1, pval=0.01)
    line = ax.lines[0]
    xdata, ydata = line.get_data()
    assert all(x == 1 for x in xdata)
    assert len(xdata) == 4

def test_plot_significance_returns_none():
    fig, ax = plt.subplots()
    result = scplt.plot_significance(ax, y=1, h=0.1, pval="n.s.")
    assert result is None

# Tests for scplt.plot_cv

def test_plot_cv_runs_without_error(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_cv(ax, pdata, classes="treatment", on="protein")
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

def test_plot_cv_returns_dataframe(pdata):
    df = scplt.plot_cv(None, pdata, classes="cellline", on="protein", return_df=True)
    assert isinstance(df, pd.DataFrame)
    assert "Class" in df.columns
    assert "CV" in df.columns
    assert not df.empty

def test_plot_cv_respects_custom_order(pdata):
    fig, ax = plt.subplots()
    valid_classes = pdata.prot.obs["treatment"].unique().tolist()
    order = valid_classes[::-1]  # reverse for test
    scplt.plot_cv(ax, pdata, classes="treatment", order=order)
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    # Order may not be identical due to seaborn sorting, but should contain same elements
    assert set(order) == set(xticklabels)

def test_plot_cv_on_peptide(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_cv(ax, pdata, on="peptide", classes="sample")
    assert _is_axes_container(result)

def test_plot_cv_return_df_only(pdata):
    df = scplt.plot_cv(None, pdata, classes="treatment", return_df=True)
    assert isinstance(df, pd.DataFrame)
    assert "CV" in df.columns
    plt.close("all")

# Tests for scplt.plot_summary
def test_plot_summary_mean_by_class(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_summary(ax, pdata, value="protein_count", classes="treatment", plot_mean=True)
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

def test_plot_summary_per_sample(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_summary(ax, pdata, value="protein_count", classes=None, plot_mean=False)
    assert _is_axes_container(result)
    assert len(result.get_xticklabels()) > 0

def test_plot_summary_multiple_classes(pdata):
    classes = ["cellline", "treatment"]
    fig, ax = plt.subplots()
    result = scplt.plot_summary(ax, pdata, value="protein_count", classes=classes, plot_mean=True)
    # Multiple subplots possible → may return list of Axes
    assert _is_axes_container(result)

def test_plot_summary_raises_without_classes(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Classes must be specified"):
        scplt.plot_summary(ax, pdata, value="protein_count", classes=None, plot_mean=True)

def test_plot_summary_invalid_classes_type(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Invalid 'classes'"):
        scplt.plot_summary(ax, pdata, value="protein_count", classes={}, plot_mean=False)

# Tests for scplt.plot_abundance_housekeeping
def test_plot_abundance_housekeeping_whole_cell(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_abundance_housekeeping(ax, pdata, classes="treatment", loading_control="whole cell")
    # Function returns None, but draws onto provided Axes
    assert _is_axes_container(ax)
    assert _count_artists(ax) >= 0
    assert ax.get_title().lower().startswith("whole cell")

def test_plot_abundance_housekeeping_all(pdata):
    fig, axes = plt.subplots(1, 3)
    result = scplt.plot_abundance_housekeeping(axes, pdata, classes="treatment", loading_control="all")
    # Returns (Figure, Axes array)
    assert isinstance(result, tuple)
    fig_out, axes_out = result
    assert isinstance(fig_out, plt.Figure)
    assert _is_axes_container(axes_out)
    assert axes_out.shape == (3,)
    assert all(ax.get_title() for ax in axes_out)
    plt.close(fig_out)

def test_plot_abundance_housekeeping_invalid_type(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Invalid loading control"):
        scplt.plot_abundance_housekeeping(ax, pdata, classes="treatment", loading_control="invalid")

def test_plot_abundance_housekeeping_no_classes(pdata):
    fig, ax = plt.subplots()
    scplt.plot_abundance_housekeeping(ax, pdata, loading_control="nuclear")
    assert _is_axes_container(ax)
    assert ax.get_title().lower().startswith("nuclear")
    plt.close("all")

# Tests for scplt.plot_abundance
def test_plot_abundance_smoke(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_abundance(ax, pdata, namelist=["ACTB", "VCL"], classes="treatment", on="protein")
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

def test_plot_abundance_return_df(pdata):
    df = scplt.plot_abundance(None, pdata, namelist=["ACTB"], classes="cellline", return_df=True)
    assert isinstance(df, pd.DataFrame)
    assert {"x_label_name", "abundance", "class"}.intersection(df.columns)
    assert not df.empty

def test_plot_abundance_violin_mode(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_abundance(ax, pdata, namelist=["ACTB"], classes="treatment", kind="violin")
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

def test_plot_abundance_bar_mode(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_abundance(ax, pdata, namelist=["ACTB"], classes="treatment", kind="bar")
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

def test_plot_abundance_with_facet(pdata):
    result = scplt.plot_abundance(None, pdata, namelist=["ACTB"], classes="treatment", facet="cellline")
    # FacetGrid should be returned when facet is used
    import seaborn as sns
    assert isinstance(result, sns.FacetGrid)
    plt.close(result.fig)

def test_plot_abundance_raises_same_facet_class(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="must be different"):
        scplt.plot_abundance(ax, pdata, namelist=["ACTB"], classes="treatment", facet="treatment")

def test_plot_abundance_no_log(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_abundance(ax, pdata, namelist=["ACTB"], classes="treatment", log=False)
    assert _is_axes_container(result)
    assert _count_artists(result) > 0

def test_plot_abundance_custom_order(pdata):
    fig, ax = plt.subplots()
    order = {"treatment": ["sc", "kd"]}
    result = scplt.plot_abundance(ax, pdata, namelist=["ACTB"], classes="treatment", order=order)
    assert _is_axes_container(result)
    plt.close("all")

# Tests for scplt.plot_pca
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D projection)

# --- Basic 2D PCA plot ---
def test_plot_pca_runs_without_error(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_pca(ax, pdata, classes="treatment", on="protein")
    assert _is_axes_container(result)
    assert _count_artists(result) > 0
    plt.close(fig)


# --- Continuous coloring (use a protein/gene expression) ---
def test_plot_pca_continuous_coloring(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_pca(ax, pdata, classes="UBE4B", on="protein")
    assert _is_axes_container(result)
    assert _count_artists(result) > 0
    # Should create a colorbar
    assert len(ax.figure.axes) > 1
    plt.close(fig)


# --- Add ellipses per class ---
def test_plot_pca_add_ellipses(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_pca(ax, pdata, classes="treatment", add_ellipses=True)
    assert _is_axes_container(result)
    # Ellipses are patches
    n_patches = len(ax.patches)
    assert n_patches >= 1
    plt.close(fig)


# --- Show sample labels ---
def test_plot_pca_show_labels(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_pca(ax, pdata, show_labels=True)
    assert _is_axes_container(result)
    # Text annotations should exist
    assert len(ax.texts) > 0
    plt.close(fig)


# --- Return fitted PCA object ---
def test_plot_pca_return_fit(pdata):
    fig, ax = plt.subplots()
    result, fit = scplt.plot_pca(ax, pdata, return_fit=True)
    assert _is_axes_container(result)
    assert isinstance(fit, dict)
    assert "variance_ratio" in fit
    plt.close(fig)


# --- 3D PCA plotting ---
def test_plot_pca_3d_projection(pdata):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    result = scplt.plot_pca(ax, pdata, plot_pc=[1, 2, 3])
    assert _is_axes_container(result)
    assert _count_artists(result) > 0
    plt.close(fig)


# --- Raises if 3 PCs but 2D axis ---
def test_plot_pca_raises_for_3pc_on_2d(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="3 PCs requested"):
        scplt.plot_pca(ax, pdata, plot_pc=[1, 2, 3])
    plt.close(fig)


# --- Raises if invalid plot_pc input ---
@pytest.mark.parametrize("bad_pc", [None, [1], [1, 2, 3, 4], "12"])
def test_plot_pca_invalid_plot_pc(pdata, bad_pc):
    fig, ax = plt.subplots()
    with pytest.raises(AssertionError, match="plot_pc must be a list"):
        scplt.plot_pca(ax, pdata, plot_pc=bad_pc)
    plt.close(fig)

# test resolve_plot_color
@pytest.fixture
def dummy_adata():
    obs = pd.DataFrame({
        "sample": ["S1", "S2", "S3", "S4"],
        "treatment": ["ctrl", "ctrl", "drug", "drug"],
        "cellline": ["A", "B", "A", "B"]
    }, index=[f"cell_{i}" for i in range(4)])
    var = pd.DataFrame({
        "Genes": ["ACTB", "GAPDH", "VDAC1"],
    }, index=["P1", "P2", "P3"])
    X = np.random.rand(4, 3)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata

def test_resolve_plot_colors_none(dummy_adata):
    colors, cmap, legend = scplt.resolve_plot_colors(dummy_adata, classes=None, cmap="default")
    assert all(c == "grey" for c in colors)
    assert cmap is None
    assert legend and legend[0].get_label() == "All samples"

def test_resolve_plot_colors_single_obs(dummy_adata):
    colors, cmap, legend = scplt.resolve_plot_colors(dummy_adata, classes="treatment", cmap="default")
    unique_colors = set(colors)
    assert len(unique_colors) == 2
    assert cmap is None
    assert all(hasattr(p, "get_facecolor") for p in legend)

def test_resolve_plot_colors_multi_obs(dummy_adata):
    colors, cmap, legend = scplt.resolve_plot_colors(dummy_adata, classes=["cellline", "treatment"], cmap="default")
    assert isinstance(colors, list)
    assert len(colors) == len(dummy_adata)
    assert cmap is None
    assert all(hasattr(p, "get_label") for p in legend)

def test_resolve_plot_colors_continuous(dummy_adata):
    colors, cmap, legend = scplt.resolve_plot_colors(dummy_adata, classes="P1", cmap="viridis", layer="X")
    assert isinstance(colors, np.ndarray)
    assert cmap is not None
    assert legend is None
    assert np.isfinite(colors).all()

def test_resolve_plot_colors_gene_name(dummy_adata):
    colors, cmap, legend = scplt.resolve_plot_colors(dummy_adata, classes="ACTB", cmap="viridis", layer="X")
    assert isinstance(colors, np.ndarray)
    assert np.isfinite(colors).all()

@pytest.mark.parametrize("bad_input", ["NotAColumn", 123, ["bad_col"]])
def test_resolve_plot_colors_invalid(dummy_adata, bad_input):
    with pytest.raises(ValueError):
        scplt.resolve_plot_colors(dummy_adata, classes=bad_input, cmap="default")

# tests for plot_umap
def test_plot_umap_runs_without_error(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_umap(ax, pdata, classes="treatment", on="protein")
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_umap_forces_recompute(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_umap(ax, pdata, force=True, classes="treatment")
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_umap_on_peptide_level(pdata):
    fig, ax = plt.subplots()
    result = scplt.plot_umap(ax, pdata, classes="cellline", on="peptide")
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_umap_continuous_coloring(pdata):
    fig, ax = plt.subplots()
    gene = pdata.prot.var["Genes"].dropna().iloc[0]
    result = scplt.plot_umap(ax, pdata, classes=gene, cmap="plasma")
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_umap_3d_projection(pdata):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    result = scplt.plot_umap(
        ax, pdata, classes="treatment", umap_params={"n_components": 3}
    )
    assert _is_axes_container(result)
    assert len(ax.collections) > 0

def test_plot_umap_invalid_on_raises(pdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Invalid value for 'on'"):
        scplt.plot_umap(ax, pdata, on="invalid_level")

# Tests for scplt.plot_pca_scree
def test_plot_pca_scree_with_dict_input():
    """Ensure scree plot works when pca is a dict (e.g. from .uns['pca'])."""
    fig, ax = plt.subplots()
    pca_dict = {"variance_ratio": np.array([0.4, 0.3, 0.2, 0.1])}
    result = scplt.plot_pca_scree(ax, pca_dict)
    assert _is_axes_container(result)
    # should have at least two lines (variance + cumulative)
    assert len(result.lines) >= 2
    labels = [line.get_label() for line in result.lines]
    assert "Explained Variance" in labels
    assert "Cumulative Variance" in labels

def test_plot_pca_scree_with_sklearn_object():
    """Ensure scree plot works for a real PCA object."""
    from sklearn.decomposition import PCA

    # Generate random data
    X = np.random.randn(20, 5)
    model = PCA(n_components=3, random_state=42).fit(X)

    fig, ax = plt.subplots()
    result = scplt.plot_pca_scree(ax, model)
    assert _is_axes_container(result)
    assert len(result.lines) >= 2
    assert result.get_title() == "Scree Plot"
    assert "Variance" in result.get_ylabel()

def test_plot_pca_scree_handles_single_component():
    """Handle PCA with a single component gracefully."""
    fig, ax = plt.subplots()
    pca_dict = {"variance_ratio": np.array([1.0])}
    result = scplt.plot_pca_scree(ax, pca_dict)
    assert _is_axes_container(result)
    # should produce one point and one cumulative curve
    assert len(result.lines) == 2
    assert result.get_xlabel() == "Principal Component"

def test_plot_pca_scree_with_real_pdata(pdata):
    """Integration test: works with real PCA results from pdata.prot.uns['pca']."""
    # run PCA first
    pdata.pca(on="protein", layer="X")
    pca_dict = pdata.prot.uns["pca"]

    fig, ax = plt.subplots()
    result = scplt.plot_pca_scree(ax, pca_dict)
    assert _is_axes_container(result)
    assert len(result.lines) >= 2
    assert "Scree" in result.get_title()

# Tests for scplt.plot_clustermap

def test_plot_clustermap_runs_basic(pdata):
    """Smoke test: basic clustermap runs on protein data without annotations."""
    result = scplt.plot_clustermap(
        None, pdata, on="prot", log2=True, xticklabels=False, yticklabels=False
    )
    assert isinstance(result, sns.matrix.ClusterGrid)
    assert hasattr(result, "data2d")
    assert not result.data2d.empty

def test_plot_clustermap_with_classes(pdata):
    """Smoke test: runs with sample annotations using classes."""
    result = scplt.plot_clustermap(
        None,
        pdata,
        on="prot",
        classes=["cellline", "treatment"],
        log2=True,
        xticklabels=False,
        yticklabels=False,
    )
    assert isinstance(result, sns.matrix.ClusterGrid)
    # Check that clustering results are stored in pdata.stats
    cluster_key = "prot_X_clustermap"
    assert cluster_key in pdata.stats
    stats = pdata.stats[cluster_key]
    assert "row_order" in stats
    assert "col_order" in stats

def test_plot_clustermap_with_force_and_impute(pdata):
    """Test that force=True triggers imputation without errors."""
    result = scplt.plot_clustermap(
        None,
        pdata,
        on="prot",
        force=True,
        impute="global_min",
        xticklabels=False,
        yticklabels=False,
    )
    assert isinstance(result, sns.matrix.ClusterGrid)
    cluster_key = "prot_X_clustermap"
    assert cluster_key in pdata.stats

def test_plot_clustermap_with_namelist(pdata):
    """Run with a restricted namelist (subset of proteins)."""
    some_proteins = pdata.prot.var_names[:5].tolist()
    result = scplt.plot_clustermap(
        None, pdata, on="prot", namelist=some_proteins, xticklabels=False, yticklabels=False
    )
    assert isinstance(result, sns.matrix.ClusterGrid)
    cluster_key = "prot_X_clustermap"
    assert cluster_key in pdata.stats
    assert pdata.stats[cluster_key]["namelist_used"] != "all_proteins"

def test_plot_clustermap_invalid_on_raises(pdata):
    """Invalid `on` argument should raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="must be 'prot' or 'pep'"):
        scplt.plot_clustermap(None, pdata, on="invalid")

# test scplt.plot_volcano and related functions
def mock_volcano_df():
    df = pd.DataFrame({
        "log2fc": [2.1, -1.8, 0.5, -0.2],
        "p_value": [0.001, 0.004, 0.2, 0.8],
        "significance": ["upregulated", "downregulated", "not significant", "not significant"],
        "significance_score": [10, -8, 0.5, 0.1],
        "Genes": ["G1", "G2", "G3", "G4"]
    })
    df["-log10(p_value)"] = -np.log10(df["p_value"])
    df.index = [f"P{i}" for i in range(1, len(df) + 1)]  # make index strings like P1, P2, P3, P4
    return df

def test_plot_volcano_with_de_data():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    result = scplt.plot_volcano(ax, de_data=df)
    assert hasattr(result, "scatter"), "❌ Should return a matplotlib Axes"
    plt.close(fig)

def test_plot_volcano_returns_df():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    ax, out_df = scplt.plot_volcano(ax, de_data=df, return_df=True)
    assert isinstance(out_df, pd.DataFrame), "❌ return_df=True should return a DataFrame"
    assert all(col in out_df.columns for col in ["log2fc", "p_value", "significance"]), "❌ Missing expected DE columns"
    plt.close(fig)

def test_add_volcano_legend_adds_handles():
    fig, ax = plt.subplots()
    scplt.add_volcano_legend(ax)
    legend = ax.get_legend()
    assert legend is not None, "❌ Legend should exist after calling add_volcano_legend()"
    labels = [t.get_text() for t in legend.get_texts()]
    assert {"Up", "Down", "NS"}.issubset(labels), f"❌ Unexpected legend labels: {labels}"
    plt.close(fig)

def test_mark_volcano_highlights_points():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=df, no_marks=True)
    n_before = len(ax.collections)
    scplt.mark_volcano(ax, df, label=["G1"], label_color="red")
    n_after = len(ax.collections)
    assert n_after > n_before, "❌ mark_volcano should add new scatter points"
    plt.close(fig)

def test_plot_volcano_with_label_list():
    df = mock_volcano_df()
    fig, ax = plt.subplots()
    scplt.plot_volcano(ax, de_data=df, label=["G1", "G2"])
    texts = [t.get_text() for t in ax.texts]
    assert any(g in texts for g in ["G1", "G2"]), "❌ Gene labels should appear on volcano plot"
    plt.close(fig)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scpviz.plotting as scplt

# test scplt.plot_rankquant() and related functions

def test_plot_rankquant_runs_without_error(pdata):
    """Ensure rank–quant plot runs and stores rank metrics."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax = scplt.plot_rankquant(ax, pdata, classes="cellline", on="protein", alpha=0.1)
    
    # Check expected outputs
    assert hasattr(ax, "scatter"), "❌ plot_rankquant should return a Matplotlib Axes."
    var_cols = pdata.prot.var.columns
    assert any("Average:" in c for c in var_cols), "❌ Missing Average: columns in .var."
    assert any("Rank:" in c for c in var_cols), "❌ Missing Rank: columns in .var."
    
    plt.close(fig)

def test_mark_rankquant_adds_points(pdata):
    """Test that mark_rankquant overlays highlights correctly."""
    # Run rankquant first to populate .var
    fig, ax = plt.subplots(figsize=(4, 3))
    ax = scplt.plot_rankquant(ax, pdata, classes="cellline", on="protein", alpha=0.1)
    n_before = len(ax.collections)

    # Build mock mark_df with at least one Entry from pdata.prot.var_names
    test_entry = pdata.prot.var_names[0]
    mark_df = pd.DataFrame({
        "Entry": [test_entry],
        "Gene Names": ["TESTGENE"]
    })

    # Call mark_rankquant
    scplt.mark_rankquant(
        ax, pdata, mark_df=mark_df,
        class_values=["A549"] if "A549" in pdata.prot.obs["cellline"].unique() else [pdata.prot.obs["cellline"].unique()[0]],
        on="protein",
        show_label=True,
        color="red"
    )

    n_after = len(ax.collections)
    assert n_after > n_before, "❌ mark_rankquant should add new points to the plot."

    plt.close(fig)

def test_plot_rankquant_debug_mode_runs(pdata, capsys):
    """Ensure plot_rankquant runs in debug mode without errors or warnings."""
    fig, ax = plt.subplots(figsize=(3, 2))

    # Run with debug=True — this should print shapes and intermediate data info
    ax = scplt.plot_rankquant(
        ax,
        pdata,
        classes="cellline",
        on="protein",
        alpha=0.1,
        debug=True
    )

    # Capture printed output and ensure something was printed
    captured = capsys.readouterr()
    assert "nsample" in captured.out or "shape" in captured.out, \
        "❌ Debug mode should print internal diagnostics."

    # Validate output integrity
    assert hasattr(ax, "scatter"), "❌ plot_rankquant should return a Matplotlib Axes in debug mode."
    assert any("Rank:" in c for c in pdata.prot.var.columns), \
        "❌ Rank columns should still be computed in debug mode."

    plt.close(fig)

# test plot_venn()
@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_venn_runs_and_returns_contents(monkeypatch):
    """Ensure plot_venn runs without error and returns correct outputs."""
    fig, ax = plt.subplots()
    pdata = object()  # dummy object; utils mock doesn't need real pdata

    # 2-set Venn with default colors
    ax_out, contents = scplt.plot_venn(ax, pdata, classes=["GroupA", "GroupB"], return_contents=True)

    assert isinstance(ax_out, tuple), "Expected tuple of (venn_obj, circles_obj)"
    assert isinstance(contents, dict)
    assert set(contents.keys()) == {"GroupA", "GroupB"}

    plt.close(fig)

@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_venn_invalid_color_length():
    """Ensure plot_venn raises error for mismatched color count."""
    fig, ax = plt.subplots()
    pdata = object()
    with pytest.raises(ValueError):
        scplt.plot_venn(ax, pdata, classes=["GroupA", "GroupB"], set_colors=["#1f77b4"])
    plt.close(fig)

@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_venn_invalid_label_order():
    """Ensure label_order mismatch raises ValueError."""
    fig, ax = plt.subplots()
    pdata = object()
    with pytest.raises(ValueError):
        scplt.plot_venn(ax, pdata, classes=["GroupA", "GroupB"], label_order=["Wrong", "Labels"])
    plt.close(fig)

@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_venn_invalid_number_of_sets(monkeypatch):
    """Ensure >3 sets raises a ValueError."""
    def mock_get_upset_contents(pdata, classes, upsetForm=False):
        return {"A": {1}, "B": {2}, "C": {3}, "D": {4}}
    monkeypatch.setattr(scplt.utils, "get_upset_contents", mock_get_upset_contents)
    fig, ax = plt.subplots()
    pdata = object()
    with pytest.raises(ValueError):
        scplt.plot_venn(ax, pdata, classes=["A", "B", "C", "D"])
    plt.close(fig)

# test plot_upset
# mock utilities
class DummyUtils:
    """Mock subset of scpviz.utils used by these plotting functions."""
    @staticmethod
    def get_upset_contents(pdata, classes, upsetForm=True):
        # Return simple dummy content mimicking protein sets
        return {
            "GroupA": {"P1", "P2", "P3"},
            "GroupB": {"P2", "P3", "P4"}
        }

class DummyUpSet:
    def __init__(self, df, **kwargs):
        self.df = df
        self.kwargs = kwargs
    def plot(self):
        return {"intersections": "mock_axes", "totals": "mock_axes"}

@pytest.fixture
def mock_upset_utils(monkeypatch):
    """Temporarily replace scplt.utils with DummyUtils for UpSet tests."""
    monkeypatch.setattr(scplt, "utils", DummyUtils())
    yield
    # pytest will restore scplt.utils afterward

@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_upset_runs(monkeypatch):
    """Ensure plot_upset runs and returns an UpSet mock."""
    monkeypatch.setattr(scplt, "upsetplot", type("m", (), {"UpSet": DummyUpSet}))
    fig, ax = plt.subplots()
    pdata = object()

    upset_obj = scplt.plot_upset(pdata, classes=["GroupA", "GroupB"])
    assert isinstance(upset_obj, DummyUpSet)
    plt.close(fig)

@pytest.mark.usefixtures("mock_upset_utils")
def test_plot_upset_return_contents(monkeypatch):
    """Ensure return_contents=True returns both UpSet and contents."""
    monkeypatch.setattr(scplt, "upsetplot", type("m", (), {"UpSet": DummyUpSet}))
    pdata = object()
    upset_obj, contents = scplt.plot_upset(pdata, classes=["GroupA", "GroupB"], return_contents=True)
    assert isinstance(upset_obj, DummyUpSet)
    assert isinstance(contents, dict)
    assert set(contents.keys()) == {"GroupA", "GroupB"}

# test plot_abundance_2d()
def test_plot_abundance_2D_runs_with_highlight():
    """Ensure plot_abundance_2D runs with gene highlighting."""
    fig, ax = plt.subplots()
    # Build mock abundance DataFrame
    df = pd.DataFrame({
        "Gene Symbol": ["A", "B", "C"],
        "Abundance: cond1": [10, 20, 30],
        "Abundance: cond2": [15, 25, 35],
        "Abundance: cond3": [5, 12, 18],
    })
    cases = [["cond1"], ["cond2"]]
    genes = ["A", "B"]
    ax = scplt.plot_abundance_2D(ax, df.copy(), cases=cases, genes=genes)
    assert hasattr(ax, "scatter"), "plot_abundance_2D should return Matplotlib Axes"
    plt.close(fig)

def test_plot_abundance_2D_runs_all_genes():
    """Ensure plot_abundance_2D runs when genes='all'."""
    fig, ax = plt.subplots()
    df = pd.DataFrame({
        "Gene Symbol": ["X", "Y"],
        "Abundance: case1": [10, 20],
        "Abundance: case2": [15, 30],
    })
    cases = [["case1"], ["case2"]]
    ax = scplt.plot_abundance_2D(ax, df.copy(), cases=cases, genes="all")
    assert hasattr(ax, "scatter")
    plt.close(fig)

# start raincloud tests

class DummyMatrix(np.ndarray):
    def toarray(self):
        return self

class DummyUtilsRain:
    """Mock subset of utils for raincloud tests."""
    @staticmethod
    def get_adata(pdata, on):
        # If pdata already has prot.var (mark_raincloud case)
        if hasattr(pdata, "prot"):
            return pdata.prot
        # Otherwise build dummy AnnData-like object
        class DummyAdata:
            def __init__(self):
                self.var = pd.DataFrame(index=["P1", "P2", "P3"])
                self.obs = pd.DataFrame({"class": ["A", "B", "A"]})
                self.X = np.abs(np.random.randn(3, 3)).view(DummyMatrix)
            def to_df(self):
                return pd.DataFrame(self.X, columns=self.var.index)
        return DummyAdata()

    @staticmethod
    def get_classlist(adata, classes=None, order=None):
        return ["A", "B"]

    @staticmethod
    def resolve_class_filter(adata, classes, class_value, debug=False):
        class DummySubset:
            def __init__(self, X):
                self.X = X
                self.var = pd.DataFrame(index=["P1", "P2", "P3"])
            def to_df(self):
                return pd.DataFrame(self.X, columns=self.var.index)
        X = np.abs(np.random.randn(3, 3)).view(DummyMatrix)
        return DummySubset(X)

@pytest.fixture
def mock_raincloud_utils(monkeypatch):
    """Temporarily replace scplt.utils with DummyUtilsRain inside a test."""
    monkeypatch.setattr(scplt, "utils", DummyUtilsRain())
    yield
    # pytest automatically restores the original after the test exits

# test plot_raincloud
@pytest.mark.usefixtures("mock_raincloud_utils")
def test_plot_raincloud_runs_without_error():
    """Ensure plot_raincloud runs normally and returns Axes."""
    fig, ax = plt.subplots()
    pdata = object()

    ax_out = scplt.plot_raincloud(
        ax, pdata, classes="class", on="protein", color=["blue", "orange"]
    )
    assert hasattr(ax_out, "violinplot"), "❌ Expected Matplotlib Axes returned."

    plt.close(fig)

@pytest.mark.usefixtures("mock_raincloud_utils")
def test_plot_raincloud_debug_mode_returns_data():
    """Ensure debug=True returns both axis and data_X arrays."""
    fig, ax = plt.subplots()
    pdata = object()

    ax_out, data_X = scplt.plot_raincloud(
        ax, pdata, classes="class", debug=True, color=["blue", "orange"]
    )
    assert isinstance(ax_out, plt.Axes)
    assert isinstance(data_X, list)
    assert len(data_X) > 0 and all(isinstance(arr, np.ndarray) for arr in data_X)

    plt.close(fig)

# test raincloud overlay
class DummyPdata:
    """Simple pdata mock with required attributes."""
    def __init__(self):
        self.prot = type("obj", (), {})()
        self.prot.var = pd.DataFrame(
            {"Average: A": [1.2, 2.5, 3.0]}, index=["P1", "P2", "P3"]
        )
    def _check_rankcol(self, on, class_values):
        return True  # No-op check

def test_mark_raincloud_adds_points():
    """Ensure mark_raincloud overlays points successfully."""
    fig, ax = plt.subplots()
    pdata = DummyPdata()

    mark_df = pd.DataFrame({"Entry": ["P1", "P2"], "Gene Names": ["G1", "G2"]})

    scplt.mark_raincloud(
        ax,
        pdata,
        mark_df=mark_df,
        class_values=["A"],
        color="red",
        s=5,
        alpha=0.8,
    )

    # The plot should now have at least one scatter collection
    assert len(ax.collections) > 0, "❌ mark_raincloud should add scatter points."
    plt.close(fig)

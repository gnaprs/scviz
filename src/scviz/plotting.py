"""
This module provides a collection of plotting utilities for visualizing
protein and peptide abundance data, quality control metrics, and results of
statistical analyses. Functions are organized into categories based on their
purpose, with paired "plot" and "mark" functions where applicable.

Functions are written to work seamlessly with the `pAnnData` object structure and metadata conventions in scviz.

## Distribution and Abundance Plots

Functions:
    plot_abundance: Violin/box/strip plots of protein or peptide abundance.
    plot_abundance_housekeeping: Plot abundance of housekeeping proteins.
    plot_rankquant: Rank abundance scatter distributions across groups.
    mark_rankquant: Highlight specific features on a rank abundance plot.
    plot_raincloud: Raincloud plot (violin + box + scatter) of distributions.
    mark_raincloud: Highlight specific features on a raincloud plot.

## Multivariate Dimension Reduction

Functions:
    plot_pca: Principal Component Analysis (PCA) scatter plot.
    plot_pca_scree: Scree plot of PCA variance explained.
    plot_umap: UMAP projection for nonlinear dimensionality reduction.
    resolve_plot_colors: Helper function for resolving PCA/UMAP colors.

## Clustering and Heatmaps

Functions:
    plot_clustermap: Clustered heatmap of proteins/peptides × samples.

## Differential Expression and Volcano Plots

Functions:
        plot_volcano: Volcano plot of differential expression results.
        mark_volcano: Highlight specific features on a volcano plot.
        add_volcano_legend: Add standard legend handles for volcano plots.

## Enrichment Plots

Functions:
    plot_enrichment_svg: Plot STRING enrichment results (forwarded from `enrichment.py`).

## Set Operations

Functions:
    plot_venn: Venn diagrams for 2 to 3 sets.
    plot_upset: UpSet diagrams for >3 sets.

## Summaries and Quality Control

Functions:
    plot_summary: Bar plots summarizing sample-level metadata (e.g., protein counts).
    plot_significance: Add significance bars to plots.
    plot_cv: Boxplots of coefficient of variation (CV) across groups.

## Notes and Tips
!!! tip
    * Most functions accept a `matplotlib.axes.Axes` as the first argument for flexible subplot integration. `ax` can be defined as such:

    ```python
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4)) # configure size as needed
    ```
    
    * "Mark" functions are designed to be used immediately after their paired "plot" functions to highlight features of interest.

---

"""

import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.collections as clt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects
from matplotlib_venn import venn2_unweighted, venn2_circles, venn3_unweighted, venn3_circles
import upsetplot
from adjustText import adjust_text
import umap.umap_ as umap
import scanpy as sc
import warnings

from scviz import utils

sns.set_theme(context='paper', style='ticks')

def get_color(resource_type: str, n=None):
    """
    Generate a list of colors, a colormap, or a palette from package defaults.

    Args:
        resource_type (str): The type of resource to generate. Options are:
            - 'colors': Return a list of hex color codes.
            - 'cmap': Return a matplotlib colormap.
            - 'palette': Return a seaborn palette.
            - 'show': Display all 7 base colors.

        n (int, optional): The number of colors or colormaps to generate.
            Required for 'colors' and 'cmap'. Colors will repeat if n > 7.

    Returns:
        colors (list of str): If ``resource_type='colors'``, a list of hex color strings. Repeats colors if n > 7.
        cmap (matplotlib.colors.LinearSegmentedColormap): If ``resource_type='cmap'``.
        palette (seaborn.color_palette): If ``resource_type='palette'``.
        None: If ``resource_type='show'``, displays the available colors.

    !!! info "Default Colors"

        The following base colors are used (hex codes):

            ['#FC9744', '#00AEE8', '#9D9D9D', '#6EDC00', '#F4D03F', '#FF0000', '#A454C7']        

        <div style="display:flex;gap:0.5em;">
            <div style="width:1.5em;height:1.5em;background:#FC9744;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#00AEE8;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#9D9D9D;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#6EDC00;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#F4D03F;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#FF0000;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#A454C7;border:1px solid #000"></div>
        </div>
            
    Example:
        Get list of 5 colors:
            ```python
            colors = get_color('colors', 5)
            ```

        <div style="display:flex;gap:0.5em;">
            <div style="width:1.5em;height:1.5em;background:#FC9744;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#00AEE8;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#9D9D9D;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#6EDC00;border:1px solid #000"></div>
            <div style="width:1.5em;height:1.5em;background:#F4D03F;border:1px solid #000"></div>
        </div>

        Get default cmap:
            ```python
            cmap = get_color('cmap', 2)
            ```
        <div style="width:150px;height:20px;background:linear-gradient(to right, white, #FC9744);border:1px solid #000"></div>
        <div style="width:150px;height:20px;background:linear-gradient(to right, white, #00AEE8);border:1px solid #000"></div>
                    
        Get default palette:
            ```python
            palette = get_color('palette')
            ```

        <div style="display:flex;gap:0.3em;">
            <div style="width:1.2em;height:1.2em;background:#FC9744;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#00AEE8;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#9D9D9D;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#6EDC00;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#F4D03F;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#FF0000;border:1px solid #000"></div>
            <div style="width:1.2em;height:1.2em;background:#A454C7;border:1px solid #000"></div>
        </div>
    """

    # --- 
    # Create a list of colors
    base_colors = ['#FC9744', '#00AEE8', '#9D9D9D', '#6EDC00', '#F4D03F', '#FF0000', '#A454C7']
    # ---

    if resource_type == 'colors':
        if n is None:
            raise ValueError("Parameter 'n' must be specified when resource_type is 'colors'")
        if n > len(base_colors):
            warnings.warn(f"Requested {n} colors, but only {len(base_colors)} available. Reusing from the start.")
        return [base_colors[i % len(base_colors)] for i in range(n)]
    
    elif resource_type == 'cmap':
        if n is None:
            n = 1  # Default to generating one colormap from the first base color
        if n > len(base_colors):
            warnings.warn(f"Requested {n} colormaps, but only {len(base_colors)} base colors. Reusing from the start.")
        cmaps = []
        for i in range(n):
            color = base_colors[i % len(base_colors)]
            cmap = mcolors.LinearSegmentedColormap.from_list(f'cmap_{i}', ['white', color])
            cmaps.append(cmap)
        return cmaps if n > 1 else cmaps[0]
    
    elif resource_type == 'palette':
        return sns.color_palette(base_colors)
    
    elif resource_type == 'show':
        # Show palette and colormaps
        fig, axs = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [1, 1]})
        
        # Format labels as "n: #HEX"
        hex_labels = [f"{i}: {mcolors.to_hex(color)}" for i, color in enumerate(base_colors)]

        # --- Palette ---
        for i, color in enumerate(base_colors):
            axs[0].bar(i, 1, color=color)
        axs[0].set_title("Base Colors (Colors and Palette)")
        axs[0].set_xticks(range(len(base_colors)))
        axs[0].set_xticklabels(hex_labels, rotation=45, ha='right')
        axs[0].set_yticks([])

        # --- Colormaps ---
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        n_colors = len(base_colors)

        for i, color in enumerate(base_colors):
            cmap = mcolors.LinearSegmentedColormap.from_list(f'cmap_{i}', ['white', color])
            axs[1].imshow(
                gradient,
                aspect='auto',
                cmap=cmap,
                extent=(i, i + 1, 0, 1)
            )

        axs[1].set_title("Colormaps")
        axs[1].set_xlim(0, n_colors)
        axs[1].set_xticks(np.arange(n_colors) + 0.5)
        axs[1].set_xticklabels(hex_labels, rotation=45, ha='right')
        axs[1].set_yticks([])

        plt.tight_layout()
        plt.show()
        return None

    else:
        raise ValueError("Invalid resource_type. Options are 'colors', 'cmap', and 'palette'")

def plot_significance(ax, y, h, x1=0, x2=1, col='k', pval='n.s.', fontsize=12):
    """
    Plot significance bars on a matplotlib axis.

    This function draws horizontal significance bars (e.g., for statistical annotations)
    between two x-positions with a label indicating the p-value or significance level.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot the significance bars.
        y (float): Vertical coordinate of the top of the bars.
        h (float): Height of the vertical ticks extending downward from `y`.
        x1 (float): X-coordinate of the first bar endpoint.
        x2 (float): X-coordinate of the second bar endpoint.
        col (str): Color of the bars.
        pval (float or str): P-value or significance label.
            
            - If a float, it is compared against thresholds (e.g., 0.05, 0.01) to assign
              significance markers (`*`, `**`, `***`).
            
            - If a string, it is directly rendered as the label.
        
        fontsize (int): Font size of the significance text.

    Returns:
        None

    !!! todo "Examples Pending"
        Add usage examples here.
    """

    # check variable type of pval
    sig = 'n.s.'
    if isinstance(pval, float):
        if pval > 0.05:
            sig = 'n.s.'
        else:
            sig = '*' * int(np.floor(-np.log10(pval)))
    else:
        sig = pval

    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
    ax.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color=col, fontsize=fontsize)

def plot_cv(ax, pdata, classes=None, layer='X', on='protein', order=None, return_df=False, **kwargs):
    """
    Generate a box-and-whisker plot for the coefficient of variation (CV).

    This function computes CV values across proteins or peptides, grouped by
    sample-level classes, and visualizes their distribution as a box plot.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData): Input pAnnData object containing protein or peptide data.
        classes (str or list of str, optional): One or more `.obs` columns to use
            for grouping samples in the plot. If None, no grouping is applied.
        layer (str): Data layer to use for CV calculation. Default is `'X'`.
        on (str): Data level to compute CV on, either `'protein'` or `'peptide'`.
        order (list, optional): Custom order of classes for plotting.
            If None, defaults to alphabetical order.
        return_df (bool): If True, returns the underlying DataFrame used for plotting.
        **kwargs: Additional keyword arguments passed to seaborn plotting functions.

    Returns:
        ax (matplotlib.axes.Axes): The axis with the plotted CV distribution.
        cv_df (pandas.DataFrame): Optional, returned if `return_df=True`.

    !!! todo "Examples Pending"
        Add usage examples here.
    """

    pdata.cv(classes = classes, on = on, layer = layer)
    adata = utils.get_adata(pdata, on)    
    classes_list = utils.get_classlist(adata, classes = classes, order = order)

    cv_data = []
    for class_value in classes_list:
        cv_col = f'CV: {class_value}'
        if cv_col in adata.var.columns:
            cv_values = adata.var[cv_col].values
            cv_data.append(pd.DataFrame({'Class': class_value, 'CV': cv_values}))

    cv_df = pd.concat(cv_data, ignore_index=True)

    # return cv_df for user to plot themselves
    if return_df:
        return cv_df
    
    sns.violinplot(x='Class', y='CV', data=cv_df, ax=ax, **kwargs)
    plt.title('Coefficient of Variation (CV) by Class')
    plt.xlabel('Class')
    plt.ylabel('CV')
    
    return ax

def plot_summary(ax, pdata, value='protein_count', classes=None, plot_mean=True, **kwargs):
    """
    Plot summary statistics of sample metadata.

    This function visualizes values from `pdata.summary` (e.g., protein count,
    peptide count, abundance) as bar plots, optionally grouped by sample-level classes.
    It supports both per-sample visualization and mean values across groups.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData): Input pAnnData object with `.summary` metadata table.
        value (str): Column in `pdata.summary` to plot. Default is `'protein_count'`.
        classes (str or list of str, optional): Sample-level classes to group by.
            - If None: plot per-sample values directly.
            
            - If str: group by the specified column, aggregating with mean if `plot_mean=True`.
            
            - If list: when multiple classes are provided, combinations of class values
              are used for grouping and subplots are created per unique value of `classes[0]`.

        plot_mean (bool): Whether to plot mean ± standard deviation by class.
            If True, `classes` must be provided. Default is True.
        **kwargs: Additional keyword arguments passed to seaborn plotting functions.

    Returns:
        ax (matplotlib.axes.Axes or list of matplotlib.axes.Axes): The axis (or 
        list of axes if subplots are created) with the plotted summary.

    Raises:
        ValueError: If `plot_mean=True` but `classes` is not specified.
        ValueError: If `classes` is invalid (not None, str, or non-empty list).

    !!! todo "Examples Pending"
        Add usage examples here.
    """

    if pdata.summary is None:
        pdata._update_summary()

    summary_data = pdata.summary.copy()

    if plot_mean:
        if classes is None:
            raise ValueError("Classes must be specified when plot_mean is True.")
        elif isinstance(classes, str):
            sns.barplot(x=classes, y=value, hue=classes, data=summary_data, errorbar='sd', ax=ax, **kwargs)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        elif isinstance(classes, list) and len(classes) > 0:
            if len(classes) == 1:
                sns.catplot(x=classes[0], y=value, data=summary_data, hue=classes[0], kind='bar', ax=ax, **kwargs)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            elif len(classes) >= 2:
                summary_data['combined_classes'] = summary_data[classes[1:]].astype(str).agg('-'.join, axis=1)

                unique_values = summary_data[classes[0]].unique()
                num_unique_values = len(unique_values)

                fig, ax = plt.subplots(nrows=num_unique_values, figsize=(10, 5 * num_unique_values))

                if num_unique_values == 1:
                    ax = [ax]

                for ax_sub, unique_value in zip(ax, unique_values):
                    subset_data = summary_data[summary_data[classes[0]] == unique_value]
                    sns.barplot(x='combined_classes', y=value, data=subset_data, hue='combined_classes', ax=ax_sub, **kwargs)
                    ax_sub.set_title(f"{classes[0]}: {unique_value}")
                    ax_sub.set_xticklabels(ax_sub.get_xticklabels(), rotation=45, ha='right')
    else:
        if classes is None:
            sns.barplot(x=summary_data.index, y=value, data=summary_data, ax=ax, **kwargs)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        elif isinstance(classes, str):
            sns.barplot(x=summary_data.index, y=value, hue=classes, data=summary_data, ax=ax, **kwargs)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        elif isinstance(classes, list) and len(classes) > 0:
            if len(classes) == 1:
                sns.barplot(x=summary_data.index, y=value, hue=classes[0], data=summary_data, ax=ax, **kwargs)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            elif len(classes) >= 2:
                summary_data['combined_classes'] = summary_data[classes[1:]].astype(str).agg('-'.join, axis=1)
                # Create a subplot for each unique value in classes[0]
                unique_values = summary_data[classes[0]].unique()
                num_unique_values = len(unique_values)
                
                fig, ax = plt.subplots(nrows=num_unique_values, figsize=(10, 5 * num_unique_values))
                
                if num_unique_values == 1:
                    ax = [ax]  # Ensure axes is iterable
                
                for ax_sub, unique_value in zip(ax, unique_values):
                    subset_data = summary_data[summary_data[classes[0]] == unique_value]
                    sns.barplot(x=subset_data.index, y=value, hue='combined_classes', data=subset_data, ax=ax_sub, **kwargs)
                    ax_sub.set_title(f"{classes[0]}: {unique_value}")
                    ax_sub.set_xticklabels(ax_sub.get_xticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()            
        else:
            raise ValueError("Invalid 'classes' parameter. It should be None, a string, or a non-empty list.")

    plt.tight_layout()

    return ax

def plot_abundance_housekeeping(ax, pdata, classes=None, loading_control='all', **kwargs):
    """
    Plot abundance of housekeeping proteins.

    This function visualizes the abundance of canonical housekeeping proteins
    as loading controls, grouped by sample-level metadata if specified.
    Different sets of proteins are supported depending on the chosen loading
    control type.

    Args:
        ax (matplotlib.axes.Axes or list of matplotlib.axes.Axes): Axis or list of axes to plot on.
            If `loading_control='all'`, must provide a list of 3 axes.
        pdata (pAnnData): Input pAnnData object.
        classes (str or list of str, optional): One or more `.obs` columns to use for grouping samples.
        loading_control (str): Type of housekeeping controls to plot. Options:

            - `'whole cell'`: GAPDH, TBCD (β-tubulin), ACTB (β-actin), VCL (vinculin), TBP (TATA-binding protein)
            
            - `'nuclear'`: COX (cytochrome c oxidase), LMNB1 (lamin B1), PCNA (proliferating cell nuclear antigen), HDAC1 (histone deacetylase 1)
            
            - `'mitochondrial'`: VDAC1 (voltage-dependent anion channel 1)
            
            - `'all'`: plots all three categories across separate subplots.

        **kwargs: Additional keyword arguments passed to seaborn plotting functions.

    Returns:
        ax (matplotlib.axes.Axes or list of matplotlib.axes.Axes):
            Axis or list of axes with the plotted protein abundances.
    Note:
        This function assumes that the specified housekeeping proteins are annotated in `.prot.var['Genes']`. Missing proteins will be skipped during plotting and may result in empty or partially filled plots.
            
    !!! example
        Plot housekeeping protein abundance for whole cell controls:
            ```python
            from scviz import plotting as scplt
            fig, ax = plt.subplots(figsize=(6,4))
            scplt.plot_abundance_housekeeping(ax, pdata, loading_control='whole cell', classes='condition')
            ```
    """


    loading_controls = {
        'whole cell': ['GAPDH', 'TBCD', 'ACTB', 'VCL', 'TBP'],
        'nuclear': ['COX', 'LMNB1', 'PCNA', 'HDAC1'],
        'mitochondrial': ['VDAC1'],
        'all': ['GAPDH', 'TBCD', 'ACTB', 'VCL', 'TBP', 'COX', 'LMNB1', 'PCNA', 'HDAC1', 'VDAC1']
    }

    # Check validity
    if loading_control not in loading_controls:
        raise ValueError(f"❌ Invalid loading control type: {loading_control}")

    # Plot all categories as subplots
    if loading_control == 'all':
        # Create 1x3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        groups = ['whole cell', 'nuclear', 'mitochondrial']
        for ax_sub, group in zip(axes, groups):
            palette = get_color('colors', n=len(loading_controls[group]))
            plot_abundance(ax_sub, pdata, namelist=loading_controls[group], classes=classes, layer='X', palette=palette, **kwargs)
            ax_sub.set_title(group.title())
        fig.suptitle("Housekeeping Protein Abundance", fontsize=14)
        return fig, axes
    else:
        palette = get_color('colors', n=len(loading_controls[loading_control]))
        plot_abundance(ax, pdata, namelist=loading_controls[loading_control], classes=classes, layer='X', palette=palette, **kwargs)
        ax.set_title(loading_control.title())

def plot_abundance(ax, pdata, namelist=None, layer='X', on='protein',
                   classes=None, return_df=False, order=None, palette=None,
                   log=True, facet=None, height=4, aspect=0.5,
                   plot_points=True, x_label='gene', kind='auto', **kwargs):
    """
    Plot abundance of proteins or peptides across samples.

    This function visualizes expression values for selected proteins or peptides
    using violin + box + strip plots, or bar plots when the number of replicates
    per group is small. Supports grouping, faceting, and custom ordering.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on. Ignored if `facet` is used.
        pdata (pAnnData): Input pAnnData object.
        namelist (list of str, optional): List of accessions or gene names to plot.
            If None, all available features are considered.
        layer (str): Data layer to use for abundance values. Default is `'X'`.
        on (str): Data level to plot, either `'protein'` or `'peptide'`.
        classes (str or list of str, optional): `.obs` column(s) to use for grouping
            samples. Determines coloring and grouping structure.
        return_df (bool): If True, returns the DataFrame of replicate and summary values.
        order (dict or list, optional): Custom order of classes. For dictionary input,
            keys are class names and values are the ordered categories.  
            Example: `order = {"condition": ["sc", "kd"]}`.
        palette (list or dict, optional): Color palette mapping groups to colors.
        log (bool): If True, apply log2 transformation to abundance values. Default is True.
        facet (str, optional): `.obs` column to facet by, creating multiple subplots.
        height (float): Height of each facet plot. Default is 4.
        aspect (float): Aspect ratio of each facet plot. Default is 0.5.
        plot_points (bool): Whether to overlay stripplot of individual samples.
        x_label (str): Label for the x-axis, either `'gene'` or `'accession'`.
        kind (str): Type of plot. Options:

            - `'auto'`: Default; uses barplot if groups have ≤ 3 samples, otherwise violin.
            - `'violin'`: Always use violin + box + strip.
            - `'bar'`: Always use barplot.

        **kwargs: Additional keyword arguments passed to seaborn plotting functions.

    Returns:
        ax (matplotlib.axes.Axes or seaborn.FacetGrid):
            The axis or facet grid containing the plot.
        df (pandas.DataFrame, optional): Returned if `return_df=True`.

    !!! example
        Plot abundance of two selected proteins:
            ```python
            from scviz import plotting as scplt
            scplt.plot_abundance(ax, pdata, namelist=['Slc12a2','Septin6'])
            ```

    """

    # Get abundance DataFrame
    df = utils.get_abundance(
        pdata, namelist=namelist, layer=layer, on=on,
        classes=classes, log=log, x_label=x_label
    )

    # --- Handle custom class ordering ---
    if classes is not None and order is not None:
        unused = set(order) - (set([classes]) if isinstance(classes, str) else set(classes))
        if unused:
            print(f"⚠️ Unused keys in `order`: {unused} (not in `classes`)")
            
        if isinstance(classes, str):
            if classes in order:
                cat_type = pd.api.types.CategoricalDtype(order[classes], ordered=True)
                df['class'] = df['class'].astype(cat_type)
        else:
            for cls in classes:
                if cls in order and cls in df.columns:
                    cat_type = pd.api.types.CategoricalDtype(order[cls], ordered=True)
                    df[cls] = df[cls].astype(cat_type)

    # --- Sort the dataframe so group order is preserved in plotting ---
    if classes is not None:
        sort_cols = ['x_label_name']
        if isinstance(classes, str):
            sort_cols.append('class')
        else:
            sort_cols.extend(classes)
        df = df.sort_values(by=sort_cols)


    # Add facet column (plotting only)
    df['facet'] = df[facet] if facet else 'all'

    if facet and classes and facet == classes:
        raise ValueError("`facet` and `classes` must be different.")

    if return_df:
        return df

    if palette is None:
        palette = get_color('palette')

    x_col = 'x_label_name'
    y_col = 'log2_abundance' if log else 'abundance'

    if kind == 'auto':
        sample_counts = df.groupby([x_col, 'class', 'facet']).size()
        kind = 'bar' if sample_counts.min() <= 3 else 'violin'

    def _plot_bar(df):
        bar_kwargs = dict(
            ci='sd',
            capsize=0.2,
            errwidth=1.5,
            palette=palette
        )
        bar_kwargs.update(kwargs)
        if facet and df['facet'].nunique() > 1:
            g = sns.FacetGrid(df, col='facet', height=height, aspect=aspect, sharey=True)
            g.map_dataframe(sns.barplot, x=x_col, y=y_col, hue='class', **bar_kwargs)
            g.set_axis_labels("Gene" if x_label == 'gene' else "Accession", "log2(Abundance)" if log else "Abundance")
            g.set_titles("{col_name}")
            g.add_legend(title='Class', frameon=True)
            return g
        else:
            if ax is None:
                fig, _ax = plt.subplots(figsize=(6, 4))
            else:
                _ax = ax
            sns.barplot(data=df, x=x_col, y=y_col, hue='class', ax=_ax, **bar_kwargs)
            handles, labels = _ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            _ax.legend(by_label.values(), by_label.keys(), title='Class', frameon=True)
            _ax.set_ylabel("log2(Abundance)" if log else "Abundance")
            _ax.set_xlabel("Gene" if x_label == 'gene' else "Accession")
            return _ax

    def _plot_violin(df):
        violin_kwargs = dict(inner="box", linewidth=1, cut=0, alpha=0.5, density_norm="width")
        violin_kwargs.update(kwargs)
        if facet and df['facet'].nunique() > 1:
            g = sns.FacetGrid(df, col='facet', height=height, aspect=aspect, sharey=True)
            g.map_dataframe(sns.violinplot, x=x_col, y=y_col, hue='class', palette=palette, **violin_kwargs)
            if plot_points:
                def _strip(data, color, **kwargs_inner):
                    sns.stripplot(data=data, x=x_col, y=y_col, hue='class', dodge=True, jitter=True,
                                  color='black', size=3, alpha=0.5, legend=False, **kwargs_inner)
                g.map_dataframe(_strip)
            g.set_axis_labels("Gene" if x_label == 'gene' else "Accession", "log2(Abundance)" if log else "Abundance")
            g.set_titles("{col_name}")
            g.add_legend(title='Class', frameon=True)
            return g
        else:
            if ax is None:
                fig, _ax = plt.subplots(figsize=(6, 4))
            else:
                _ax = ax
            sns.violinplot(data=df, x=x_col, y=y_col, hue='class', palette=palette, ax=_ax, **violin_kwargs)
            if plot_points:
                sns.stripplot(data=df, x=x_col, y=y_col, hue='class', dodge=True, jitter=True,
                              color='black', size=3, alpha=0.5, legend=False, ax=_ax)
            handles, labels = _ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            _ax.legend(by_label.values(), by_label.keys(), title='Class', frameon=True)
            _ax.set_ylabel("log2(Abundance)" if log else "Abundance")
            _ax.set_xlabel("Gene" if x_label == 'gene' else "Accession")
            return _ax

    return _plot_bar(df) if kind == 'bar' else _plot_violin(df)

def plot_pca(ax, pdata, classes=None, layer="X", on='protein',
             cmap='default', s=20, alpha=.8, plot_pc=[1, 2],
             pca_params=None, force=False, basis='X_pca',
             show_labels=False, label_column=None,
             add_ellipses=False, ellipse_kwargs=None, return_fit=False):
    """
    Plot principal component analysis (PCA) of protein or peptide abundance.

    This function computes or reuses PCA of abundance values and visualizes
    samples in a scatter plot colored by metadata or feature expression.
    Supports both 2D and 3D plotting, with optional labels and confidence ellipses.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on. Must be 3D if plotting 3 PCs.
        pdata (pAnnData): Input pAnnData object with `.prot`, `.pep`, and `.summary`.
        classes (str, list of str, or None): Coloring scheme.
            
            - None: plot all samples in grey.
            
            - str: an `.obs` column (e.g. `"treatment"`) or a gene/protein (e.g. `"UBE4B"`).
            
            - list of str: combine multiple `.obs` columns (e.g. `["cellline", "treatment"]`).

        layer (str): Data layer to use. Default is `"X"`.
        on (str): Data level to plot, either `"protein"` or `"peptide"`. Default is `"protein"`.
        cmap (str, list, or matplotlib colormap): Colormap for point coloring.
            
            - `"default"`: uses `get_color()` scheme.
            
            - list of colors: categorical mapping for `classes`.
            
            - colormap name or object: continuous coloring for expression values.

        s (float): Scatter dot size. Default is 20.
        alpha (float): Point opacity. Default is 0.8.
        plot_pc (list of int): Principal components to plot, e.g. `[1, 2]` or `[1, 2, 3]`.
        pca_params (dict, optional): Additional parameters for `sklearn.decomposition.PCA`.
        force (bool): If True, recompute PCA even if it is already cached.
        basis (str): PCA basis to use. Defaults to `X_pca`, alternatives include `X_pca_harmony` after running pdata.harmony(batch="<key>").
        show_labels (bool or list): Whether to label points.
            
            - False: no labels.
            
            - True: label all samples.
            
            - list: label only specified samples.

        label_column (str, optional): Column in `.summary` to use as label source.
            Overrides sample names if provided.
        add_ellipses (bool): If True, overlay confidence ellipses per class (2D only).
            Ellipses represent a 95% confidence region under a bivariate Gaussian assumption.
        ellipse_kwargs (dict, optional): Additional keyword arguments for the ellipse patch.
        return_fit (bool): If True, also return the fitted PCA object.

    Returns:
        ax (matplotlib.axes.Axes): Axis containing the PCA scatter plot.

        pca (sklearn.decomposition.PCA): The fitted PCA object.

    Note:
        PCA results are cached in `pdata.uns["pca"]` and reused across plotting calls.
        To force recalculation (e.g., after filtering or normalization), set `force=True`.

    Example:
        Basic usage in grey:
            ```python
            plot_pca(ax, pdata)
            ```

        Color by categorical class:
            ```python
            plot_pca(ax, pdata, classes="treatment")
            ```

        Combine multiple classes:
            ```python
            plot_pca(ax, pdata, classes=["cellline", "treatment"])
            ```

        Color by protein expression:
            ```python
            plot_pca(ax, pdata, classes="UBE4B")
            ```

        Label all samples:
            ```python
            plot_pca(ax, pdata, show_labels=True)
            ```

        Label with custom column:
            ```python
            plot_pca(ax, pdata, show_labels=True, label_column="short_name")
            ```

        Add confidence ellipses:
            ```python
            plot_pca(ax, pdata, classes="treatment", add_ellipses=True)
            ```

        Customize ellipse appearance:
            ```python
            plot_pca(
                ax, pdata, classes="treatment", add_ellipses=True,
                ellipse_kwargs={"alpha": 0.1, "lw": 2}
            )
            ```
    """
    from matplotlib.patches import Ellipse

    def plot_confidence_ellipse(x, y, ax, n_std=2.4477, facecolor='none', edgecolor='black', alpha=0.2, **kwargs):
        if x.size <= 2:
            return
        cov = np.cov(x, y)
        if np.linalg.matrix_rank(cov) < 2:
            return
        mean_x, mean_y = np.mean(x), np.mean(y)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        width, height = 2 * n_std * np.sqrt(vals)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle,
                          facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, lw=1.5, **kwargs)
        ax.add_patch(ellipse)

    # Validate PCA dimensions
    assert isinstance(plot_pc, list) and len(plot_pc) in [2, 3], "plot_pc must be a list of 2 or 3 PCs."
    if len(plot_pc) == 3:
        assert ax.name == '3d', "3 PCs requested — ax must be a 3D projection"

    pc_x, pc_y = plot_pc[0] - 1, plot_pc[1] - 1
    pc_z = plot_pc[2] - 1 if len(plot_pc) == 3 else None

    adata = utils.get_adata(pdata, on)

    default_pca_params = {'n_comps': min(len(adata.obs_names), len(adata.var_names)) - 1, 'random_state': 42}
    pca_param = {**default_pca_params, **(pca_params or {})}

    if basis != "X_pca":
        # User-specified alternative basis (e.g. Harmony, ICA)
        if basis not in adata.obsm:
            raise KeyError(f"{utils.format_log_prefix('error',2)} Custom PCA basis '{basis}' not found in adata.obsm.")
    else:
        # Standard PCA case
        if "X_pca" not in adata.obsm or force:
            print(f"{utils.format_log_prefix('info')} Computing PCA (force={force})...")
            pdata.pca(on=on, layer=layer, **pca_param)
        else:
            print(f"{utils.format_log_prefix('info')} Using existing PCA embedding.")

    # --- Select PCA basis for plotting ---
    X_pca = adata.obsm[basis] if basis in adata.obsm else adata.obsm["X_pca"]
    pca = adata.uns["pca"]

    # Get colors
    color_mapped, cmap_resolved, legend_elements = resolve_plot_colors(adata, classes, cmap, layer=layer)

    # Plot
    if len(plot_pc) == 2:
        ax.scatter(X_pca[:, pc_x], X_pca[:, pc_y], c=color_mapped, cmap=cmap_resolved, s=s, alpha=alpha)
        ax.set_xlabel(f'PC{pc_x+1} ({pca["variance_ratio"][pc_x]*100:.2f}%)')
        ax.set_ylabel(f'PC{pc_y+1} ({pca["variance_ratio"][pc_y]*100:.2f}%)')

        # Add colorbar if using continuous color (i.e., abundance coloring)
        if isinstance(color_mapped, np.ndarray) and cmap_resolved is not None:
            norm = mcolors.Normalize(vmin=np.min(color_mapped), vmax=np.max(color_mapped))
            sm = cm.ScalarMappable(cmap=cmap_resolved, norm=norm)
            sm.set_array([])
            cb = ax.figure.colorbar(sm, ax=ax, pad=0.01)
            cb.set_label(classes if isinstance(classes, str) else "Abundance", fontsize=9)

        if add_ellipses and isinstance(classes, (str, list)) and all(c in adata.obs.columns for c in (classes if isinstance(classes, list) else [classes])):
            ellipse_kwargs = ellipse_kwargs.copy() if ellipse_kwargs else {}
            y = utils.get_samplenames(adata, classes)
            df_coords = pd.DataFrame(X_pca[:, [pc_x, pc_y]], columns=["PC1", "PC2"], index=adata.obs_names)
            df_coords['class'] = y
            for cls in df_coords['class'].unique():
                sub = df_coords[df_coords['class'] == cls]
                color_series = pd.Series(color_mapped, index=adata.obs_names)
                color = color_series[df_coords['class'] == cls].iloc[0]

                kwargs = ellipse_kwargs.copy() if ellipse_kwargs else {}
                kwargs["facecolor"] = color
                kwargs["edgecolor"] = color

                plot_confidence_ellipse(sub['PC1'].values, sub['PC2'].values, ax=ax, **kwargs)

    elif len(plot_pc) == 3:
        ax.scatter(X_pca[:, pc_x], X_pca[:, pc_y], X_pca[:, pc_z], c=color_mapped, cmap=cmap_resolved, s=s, alpha=alpha)
        ax.set_xlabel(f'PC{pc_x+1} ({pca["variance_ratio"][pc_x]*100:.2f}%)')
        ax.set_ylabel(f'PC{pc_y+1} ({pca["variance_ratio"][pc_y]*100:.2f}%)')
        ax.set_zlabel(f'PC{pc_z+1} ({pca["variance_ratio"][pc_z]*100:.2f}%)')

        # Add colorbar if using continuous color (i.e., abundance coloring)
        if isinstance(color_mapped, np.ndarray) and cmap_resolved is not None:
            norm = mcolors.Normalize(vmin=np.min(color_mapped), vmax=np.max(color_mapped))
            sm = cm.ScalarMappable(cmap=cmap_resolved, norm=norm)
            sm.set_array([])
            cb = ax.figure.colorbar(sm, ax=ax, pad=0.01)
            cb.set_label(classes if isinstance(classes, str) else "Abundance", fontsize=9)

    # Labels
    if show_labels:
        show_set = set(show_labels) if isinstance(show_labels, list) else set(adata.obs_names)
        label_series = pdata.summary[label_column] if label_column and label_column in pdata.summary.columns else adata.obs_names
        for i, sample in enumerate(adata.obs_names):
            if sample in show_set:
                label = label_series[i] if i < len(label_series) else sample
                pos = X_pca[i, [pc_x, pc_y, pc_z][:len(plot_pc)]]
                if len(pos) == 2:
                    ax.text(pos[0], pos[1], str(label), fontsize=8, ha='right', va='bottom')
                elif len(pos) == 3:
                    ax.text(pos[0], pos[1], pos[2], str(label), fontsize=8)
        if not label_column and max(len(str(n)) for n in label_series) > 20:
            print("[plot_pca] Warning: Labels are long. Consider using label_column='your_column'.")

    if legend_elements:
        if classes is None:
            legend_title = None  # no title if no classes
        elif isinstance(classes, list):
            legend_title = "/".join(c.capitalize() for c in classes)
        else:
            legend_title = str(classes).capitalize()

        ax.legend(handles=legend_elements,
                title=legend_title,
                loc='best',
                frameon=False)

    if return_fit:
        return ax, pca
    else:
        return ax

def resolve_plot_colors(adata, classes, cmap, layer="X"):
    """
    Resolve colors for PCA or abundance plots.

    This helper function determines how samples should be colored in plotting
    functions based on categorical or continuous class values. It returns mapped
    color values, a colormap (if applicable), and legend handles.

    Args:
        adata (anndata.AnnData): AnnData object (protein or peptide level).
        classes (str): Class used for coloring. Can be:
            
            - An `.obs` column name (categorical or continuous).
            
            - A gene or protein identifier, in which case coloring is based
              on abundance values from the specified `layer`.

        cmap (str, list, or matplotlib colormap): Colormap to use.
            
            - `"default"`: uses `get_color()` scheme.
            
            - list of colors: categorical mapping.
            
            - colormap name or object: continuous mapping.

        layer (str): Data layer to extract abundance values from when `classes`
            is a gene/protein. Default is `"X"`.

    Returns:      
        color_mapped (array-like): Values mapped to colors for plotting.
        cmap_resolved (matplotlib colormap or None): Colormap object for continuous coloring; None if categorical.
        legend_elements (list or None): Legend handles for categorical coloring; None if continuous.

    """

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import numpy as np

    legend_elements = None

    # Case 1: No coloring, all grey
    if classes is None:
        color_mapped = ['grey'] * len(adata)
        legend_elements = [mpatches.Patch(color='grey', label='All samples')]
        return color_mapped, None, legend_elements

    # Case 2: Single categorical column from obs
    elif isinstance(classes, str) and classes in adata.obs.columns:
        y = utils.get_samplenames(adata, classes)
        class_labels = sorted(set(y))
        if cmap == 'default':
            palette = get_color('colors', n=len(class_labels))
            color_dict = {c: palette[i] for i, c in enumerate(class_labels)}
        elif isinstance(cmap, list):
            color_dict = {c: cmap[i] for i, c in enumerate(class_labels)}
        elif isinstance(cmap, dict):
            color_dict = cmap
        else:
            cmap_obj = cm.get_cmap(cmap)
            palette = [mcolors.to_hex(cmap_obj(i / max(len(class_labels) - 1, 1))) for i in range(len(class_labels))]
            color_dict = {c: palette[i] for i, c in enumerate(class_labels)}
        color_mapped = [color_dict[val] for val in y]
        legend_elements = [mpatches.Patch(color=color_dict[c], label=c) for c in class_labels]
        return color_mapped, None, legend_elements

    # Case 3: Multiple categorical columns from obs (combined class)
    elif isinstance(classes, list) and all(c in adata.obs.columns for c in classes):
        y = utils.get_samplenames(adata, classes)
        class_labels = sorted(set(y))
        if cmap == 'default':
            palette = get_color('colors', n=len(class_labels))
            color_dict = {c: palette[i] for i, c in enumerate(class_labels)}
        elif isinstance(cmap, list):
            color_dict = {c: cmap[i] for i, c in enumerate(class_labels)}
        elif isinstance(cmap, dict):
            color_dict = cmap
        else:
            cmap_obj = cm.get_cmap(cmap)
            palette = [mcolors.to_hex(cmap_obj(i / max(len(class_labels) - 1, 1))) for i in range(len(class_labels))]
            color_dict = {c: palette[i] for i, c in enumerate(class_labels)}
        color_mapped = [color_dict[val] for val in y]
        legend_elements = [mpatches.Patch(color=color_dict[c], label=c) for c in class_labels]
        return color_mapped, None, legend_elements

    # Case 4: Continuous coloring by protein abundance (accession)
    elif isinstance(classes, str) and classes in adata.var_names:
        X = adata.layers[layer] if layer in adata.layers else adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        idx = list(adata.var_names).index(classes)
        color_mapped = X[:, idx]
        if cmap == 'default':
            cmap = 'viridis'
        cmap = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap

        # Add default colorbar handling for abundance-based coloring
        norm = mcolors.Normalize(vmin=color_mapped.min(), vmax=color_mapped.max())
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # required for colorbar

        return color_mapped, cmap, None

    # Case 5: Gene name (mapped to accession)
    elif isinstance(classes, str):
        if "Genes" in adata.var.columns:
            gene_map = adata.var["Genes"].to_dict()
            match = [acc for acc, gene in gene_map.items() if gene == classes]
            if match:
                return resolve_plot_colors(adata, match[0], cmap, layer)
        raise ValueError("Invalid classes input. Must be None, a protein in var_names, or an obs column/list.")

    else:
        raise ValueError("Invalid classes input.")

# NOTE: STRING enrichment plots live in enrichment.py, not here.
# This function is re-documented here for discoverability.
def plot_enrichment_svg(*args, **kwargs):
    """
    Plot STRING enrichment results as an SVG figure.

    This is a wrapper that redirects to the implementation in `enrichment.py`
    for convenience and discoverability.

    Args:
        *args: Positional arguments passed to `scviz.enrichment.plot_enrichment_svg`.
        **kwargs: Keyword arguments passed to `scviz.enrichment.plot_enrichment_svg`.

    Returns:
        svg (SVG): SVG figure object.

    See Also:
        scviz.enrichment.plot_enrichment_svg
    """
    from .enrichment import plot_enrichment_svg as actual_plot
    return actual_plot(*args, **kwargs)

def plot_umap(ax, pdata, classes = None, layer = "X", on = 'protein', cmap='default', s=20, alpha=.8, umap_params={}, text_size = 10, force = False, return_fit=False):
    """
    Plot UMAP projection of protein or peptide abundance data.

    This function projects the data using UMAP and colors samples based on
    metadata or abundance of a specific feature. Supports categorical or
    continuous coloring, with automatic handling of legends and colormaps.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on (must be 3D if n_components=3).
        pdata (scviz.pAnnData): The pAnnData object containing .prot, .pep, and .summary.
        classes (str or list of str or None): 
            - None: all samples plotted in grey
            - str: column in `.obs` or a gene/protein to color by
            - list of str: combine multiple `.obs` columns (e.g., ['cellline', 'day'])
        layer (str): Data layer to use for UMAP input (default: "X").
        on (str): Whether to use 'protein' or 'peptide' data (default: 'protein').
        cmap (str, list, or dict): 
            - 'default': use internal color scheme
            - list: list of colors, assigned to class labels in sorted order
            - dict: {label: color} mapping
            - str: continuous colormap name (e.g., 'viridis') for abundance coloring
        s (float): Marker size (default: 20).
        alpha (float): Marker opacity (default: 0.8).
        umap_params (dict): Parameters to pass to UMAP (e.g., 'min_dist', 'metric').
        text_size (int): Font size for axis labels and legend (default: 10).
        force (bool): If True, re-compute UMAP even if results already exist.
        return_fit (bool): If True, return the fitted UMAP object along with the axis.

    Returns:
        ax (matplotlib.axes.Axes): The axis with the UMAP plot.
        fit_umap (umap.UMAP): The fitted UMAP object.

    Raises:
        AssertionError: If 'n_components' is 3 and the axis is not 3D.

    Example:
        Plot by treatment group with default palette:
            ```python
            plot_umap(ax, pdata, classes='treatment')
            ```

        Plot by protein abundance (continuous coloring):
            ```python
            plot_umap(ax, pdata, classes='P12345', cmap='plasma')
            ```

        Plot with custom palette:
            ```python
            custom_palette = {'ctrl': '#CCCCCC', 'treated': '#E41A1C'}
            plot_umap(ax, pdata, classes='group', cmap=custom_palette)
            ```
    """
    default_umap_params = {'n_components': 2, 'random_state': 42}
    umap_param = {**default_umap_params, **(umap_params if umap_params else {})}
    
    if umap_param['n_components'] == 3:
        assert ax.name == '3d', "The ax must be a 3D projection, please define projection='3d'"

    if on == 'protein':
        adata = pdata.prot
    elif on == 'peptide':
        adata = pdata.pep
    else:
        raise ValueError("Invalid value for 'on'. Options are 'protein' or 'peptide'.")
 
    if force == False:
        if 'X_umap' in adata.obsm.keys():
            print(f'{utils.format_log_prefix("warn")} UMAP already exists in {on} data, using existing UMAP. Run with `force=True` to recompute.')
        else:
            pdata.umap(on=on, layer=layer, **umap_param)
    else:
        print(f'UMAP calculation forced, re-calculating UMAP')
        pdata.umap(on=on, layer=layer, **umap_param)

    Xt = adata.obsm['X_umap']
    umap = adata.uns['umap']

    color_mapped, cmap_resolved, legend_elements = resolve_plot_colors(adata, classes, cmap, layer=layer)

    if umap_param['n_components'] == 1:
        ax.scatter(Xt[:,0], range(len(Xt)), c=color_mapped, cmap=cmap_resolved, s=s, alpha=alpha)
        ax.set_xlabel('UMAP 1', fontsize=text_size)
    elif umap_param['n_components'] == 2:
        ax.scatter(Xt[:,0], Xt[:,1], c=color_mapped, cmap=cmap_resolved, s=s, alpha=alpha)
        ax.set_xlabel('UMAP 1', fontsize=text_size)
        ax.set_ylabel('UMAP 2', fontsize=text_size)
    elif umap_param['n_components'] == 3:
        ax.scatter(Xt[:,0], Xt[:,1], Xt[:,2], c=color_mapped, cmap=cmap_resolved, s=s, alpha=alpha)
        ax.set_xlabel('UMAP 1', fontsize=text_size)
        ax.set_ylabel('UMAP 2', fontsize=text_size)
        ax.set_zlabel('UMAP 3', fontsize=text_size)

    if legend_elements:
        if classes is None:
            legend_title = None
        elif isinstance(classes, list):
            legend_title = "/".join(c.capitalize() for c in classes)
        else:
            legend_title = str(classes).capitalize()
        ax.legend(handles=legend_elements, title=legend_title, loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=text_size)

    if return_fit:
        return ax, umap
    else:
        return ax

def plot_pca_scree(ax, pca):
    """
    Plot a scree plot of explained variance from PCA.

    This function visualizes the proportion of variance explained by each
    principal component as a bar chart, helping to assess how many PCs are
    meaningful.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot the scree plot.

        pca (sklearn.decomposition.PCA or dict): The fitted PCA object, or a
            dictionary from `.uns` with key `"variance_ratio"`.

    Returns:
        ax (matplotlib.axes.Axes): Axis containing the scree plot.

    Example:
        Basic usage with fitted PCA, first run PCA:
            ```python
            import matplotlib.pyplot as plt
            from scviz import plotting as scplt
            fig, ax = plt.subplots()
            ax, pca = scplt.plot_pca(ax, pdata, classes=["cellline", "treatment"], plot_pc=[1, 2])  # run PCA and plot
            ax = scplt.plot_pca_scree(ax, pca)  # scree plot
            ```

        If PCA has already been run, use cached PCA results from `.uns`:
            ```python
            scplt.plot_pca_scree(ax, pdata.prot.uns["pca"])
            ```
    """
    if isinstance(pca, dict):
        variance_ratio = np.array(pca["variance_ratio"])
        n_components = len(variance_ratio)
    else:
        variance_ratio = pca.explained_variance_ratio_
        n_components = pca.n_components_

    PC_values = np.arange(1, n_components + 1)
    cumulative = np.cumsum(variance_ratio)

    ax.plot(PC_values, variance_ratio, 'o-', linewidth=2, label='Explained Variance', color='blue')
    ax.plot(PC_values, cumulative, 'o--', linewidth=2, label='Cumulative Variance', color='gray')
    ax.set_title('Scree Plot')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    
    return ax

def plot_clustermap(ax, pdata, on='prot', classes=None, layer="X", x_label='accession', namelist=None, lut=None, log2=True,
                    cmap="coolwarm", figsize=(6, 10), force=False, impute=None, order=None, **kwargs):
    """
    Plot a clustered heatmap of proteins or peptides by samples.

    This function creates a hierarchical clustered heatmap (features × samples)
    with optional column annotations from sample-level metadata. Supports
    custom annotation colors, log2 transformation, and missing value imputation.

    Args:
        ax (matplotlib.axes.Axes): Unused; included for API compatibility.
        pdata (pAnnData): Input pAnnData object.
        on (str): Data level to plot, either `"prot"` or `"pep"`. Default is `"prot"`.
        classes (str or list of str, optional): One or more `.obs` columns to
            annotate samples in the heatmap.
        layer (str): Data layer to use. Defaults to `"X"`.
        x_label (str): Row label mode, either `"accession"` or `"gene"`. Used
            for mapping `namelist`.
        namelist (list of str, optional): Subset of accessions or gene names to plot.
            If None, all rows are included.
        lut (dict, optional): Nested dictionary of `{class_name: {label: color}}`
            controlling annotation bar colors. Missing entries fall back to
            default palettes. See the note 'lut example' below.
        log2 (bool): Whether to log2-transform the abundance matrix. Default is True.
        cmap (str): Colormap for heatmap. Default is `"coolwarm"`.
        figsize (tuple): Figure size in inches. Default is `(6, 10)`.
        force (bool): If True, imputes missing values instead of dropping rows
            with NaNs.
        impute (str, optional): Imputation strategy used when `force=True`.
            
            - `"row_min"`: fill NaNs with minimum value of that protein row.
            - `"global_min"`: fill NaNs with global minimum value of the matrix.

        order (dict, optional): Custom order for categorical annotations.
            Example: `{"condition": ["kd", "sc"], "cellline": ["AS", "BE"]}`.
        **kwargs: Additional keyword arguments passed to `seaborn.clustermap`.

            Common options include:
            
            - `z_score (int)`: Normalize rows (0, features) or columns (1, samples).
            - `standard_scale (int)`: Scale rows or columns to unit variance.
            - `center (float)`: Value to center colormap on (e.g. 0 with `z_score`).
            - `col_cluster (bool)`: Cluster columns (samples). Default is False.
            - `row_cluster (bool)`: Cluster rows (features). Default is True.
            - `linewidth (float)`: Grid line width between cells.
            - `xticklabels` / `yticklabels` (bool): Show axis tick labels.
            - `colors_ratio (tuple)`: Proportion of space allocated to annotation bars.

    Returns:
        g (seaborn.matrix.ClusterGrid): The seaborn clustermap object.

    !!! note "lut example"
        Example of a custom lookup table for annotation colors:
            ```python
            lut = {
                "cellline": {
                    "AS": "#e41a1c",
                    "BE": "#377eb8"
                },
                "condition": {
                    "kd": "#4daf4a",
                    "sc": "#984ea3"
               }
            }
            ```

    !!! todo "Examples Pending"
        Add usage examples here.
    """
    # --- Step 1: Extract data ---
    if on not in ("prot", "pep"):
        raise ValueError(f"`on` must be 'prot' or 'pep', got '{on}'")
    
    if namelist is not None:
        df_abund = utils.get_abundance(
        pdata, namelist=namelist, layer=layer, on=on,
        classes=classes, log=log2, x_label=x_label)
        
        pivot_col = "log2_abundance" if log2 else "abundance"
        row_index = "gene" if x_label == "gene" else "accession"
        df = df_abund.pivot(index=row_index, columns="cell", values=pivot_col)
    
    else:
        adata = pdata.prot if on == 'prot' else pdata.pep
        X = adata.layers[layer] if layer in adata.layers else adata.X
        data = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        df = pd.DataFrame(data.T, index=adata.var_names, columns=adata.obs_names)
        if log2:
            with np.errstate(divide='ignore', invalid='ignore'):
                df = np.log2(df)
                df[df == -np.inf] = np.nan

    # --- Handle missing values ---
    nan_rows = df.index[df.isna().any(axis=1)].tolist()
    if nan_rows:
        if not force:
            print(f"Warning: {len(nan_rows)} proteins contain missing values and will be excluded: {nan_rows}")
            print("To include them, rerun with force=True and impute='row_min' or 'global_min'.")
            df = df.drop(index=nan_rows)
        else:
            print(f"{len(nan_rows)} proteins contain missing values: {nan_rows}.\nImputing using strategy: '{impute}'")
            if impute == "row_min":
                global_min = df.min().min()
                df = df.apply(lambda row: row.fillna(row.min() if not np.isnan(row.min()) else global_min), axis=1)
            elif impute == "global_min":
                df = df.fillna(df.min().min())
            else:
                raise ValueError("`impute` must be either 'row_min' or 'global_min' when force=True.")

    # --- Step 2: Column annotations ---
    col_colors = None
    legend_handles, legend_labels = [], []

    if classes is not None:
        if isinstance(classes, str):
            sample_labels = utils.get_samplenames(adata, classes)
            annotations = pd.DataFrame({classes: sample_labels}, index=adata.obs_names)
        else:
            sample_labels = utils.get_samplenames(adata, classes)
            split_labels = [[part.strip() for part in s.split(",")] for s in sample_labels]
            annotations = pd.DataFrame(split_labels, index=adata.obs_names, columns=classes)

        # Optional: apply custom category order from `order` dict
        if order is not None and isinstance(order, dict):
            for col in classes:
                if col in annotations.columns and col in order:
                    cat_type = pd.api.types.CategoricalDtype(order[col], ordered=True)
                    annotations[col] = annotations[col].astype(cat_type)
            unused_keys = set(order) - set(classes)
            if unused_keys:
                print(f"⚠️ Unused keys in `order`: {unused_keys} (not present in `classes`)")

        # Sort columns (samples) by class hierarchy
        sort_order = annotations.sort_values(by=classes).index
        df = df[sort_order]
        annotations = annotations.loc[sort_order]

        if lut is None:
            lut = {}

        full_lut = {}
        for col in annotations.columns:
            unique_vals = sorted(annotations[col].dropna().unique())
            user_colors = lut.get(col, {})
            missing_vals = [v for v in unique_vals if v not in user_colors]
            fallback_palette = sns.color_palette(n_colors=len(missing_vals))
            fallback_colors = dict(zip(missing_vals, fallback_palette))
            full_lut[col] = {**user_colors, **fallback_colors}

            unmatched = set(user_colors) - set(unique_vals)
            if unmatched:
                print(f"Warning: The following labels in `lut['{col}']` are not found in the data: {sorted(unmatched)}")

        col_colors = annotations.apply(lambda col: col.map(full_lut[col.name]))

        # Legend handles
        for col in annotations.columns:
            legend_handles.append(mpatches.Patch(facecolor="none", edgecolor="none", label=col))  # header
            for label, color in full_lut[col].items():
                legend_handles.append(mpatches.Patch(facecolor=color, edgecolor="black", label=label))
            legend_labels.extend([col] + list(full_lut[col].keys()))

    # --- Step 3: Clustermap defaults (user-overridable) ---
    col_cluster = kwargs.pop("col_cluster", False)
    row_cluster = kwargs.pop("row_cluster", True)
    linewidth = kwargs.pop("linewidth", 0)
    yticklabels = kwargs.pop("yticklabels", False)
    xticklabels = kwargs.pop("xticklabels", False)
    colors_ratio = kwargs.pop("colors_ratio", (0.03, 0.02))
    if kwargs.get("z_score", None) == 0:
        zero_var_rows = df.var(axis=1) == 0
        if zero_var_rows.any():
            dropped = df.index[zero_var_rows].tolist()
            print(f"⚠️ {len(dropped)} proteins have zero variance and will be dropped due to z_score=0: {dropped}")
            df = df.drop(index=dropped)

    # --- Step 4: Plot clustermap ---
    try:
        g = sns.clustermap(df,
                        cmap=cmap,
                        col_cluster=col_cluster,
                        row_cluster=row_cluster,
                        col_colors=col_colors,
                        figsize=figsize,
                        xticklabels=xticklabels,
                        yticklabels=yticklabels,
                        linewidth=linewidth,
                        colors_ratio=colors_ratio,
                        **kwargs)
    except Exception as e:
        print(f"Error occurred while creating clustermap: {e}")
        return df

    # --- Step 5: Column annotation legend ---
    if classes is not None:
        g.ax_col_dendrogram.legend(legend_handles, legend_labels,
                                   title=None,
                                   bbox_to_anchor=(0.5, 1.15),
                                   loc="upper center",
                                   ncol=len(classes),
                                   handletextpad=0.5,
                                   columnspacing=1.5,
                                   frameon=False)
        
    # --- Step 6: Row label remapping ---
    if x_label == "gene" and xticklabels:
        _ , prot_map = pdata.get_gene_maps(on='protein' if on == 'prot' else 'peptide')
        row_labels = [prot_map.get(row, row) for row in g.data2d.index]
        g.ax_heatmap.set_yticklabels(row_labels, rotation=0)

    # --- Step 8: Store clustering results ---
    cluster_key  = f"{on}_{layer}_clustermap"
    row_order = list(g.data2d.index)
    row_indices = g.dendrogram_row.reordered_ind

    pdata.stats[cluster_key]  = {
        "row_order": row_order,
        "row_indices": row_indices,
        "row_labels": x_label,   # 'accession' or 'gene'
        "namelist_used": namelist if namelist is not None else "all_proteins",
        "col_order": list(g.data2d.columns),
        "col_indices": g.dendrogram_col.reordered_ind if g.dendrogram_col else None,
        "row_linkage": g.dendrogram_row.linkage,  # <--- NEW
        "col_linkage": g.dendrogram_col.linkage if g.dendrogram_col else None,
    }

    return g

# def plot_heatmap(ax, pdata, classes=None, layer="X", cmap=cm.get_cmap('seismic'), norm_values=[4,5.5,7], linewidth=.5, annotate=True, square=False, cbar_kws = {'label': 'Abundance (AU)'}):
#     """
#     Plot annotated heatmap of protein abundance data.

#     Parameters:
#     ax (matplotlib.axes.Axes): The axes on which to plot the heatmap.
#     pdata (scviz.pdata): The input pdata object.
#     classes (str or list of str, optional): Class column(s) to group samples. If None, all samples are included.
#     cmap (matplotlib.colors.Colormap): The colormap to use for the heatmap.
#     norm_values (list): The low, mid, and high values used to set colorbar scale. Can be assymetric.
#     linewidth (float): Plot linewidth.
#     annotate (bool): Annotate each heatmap entry with numerical value. True by default.
#     square (bool): Make heatmap square. False by default.
#     cbar_kws (dict): Pass-through keyword arguments for the colorbar. See `matplotlib.figure.Figure.colorbar()` for more information.

#     Returns:
#     ax (matplotlib.axes.Axes): The axes with the plotted heatmap.
#     """
#     # get the abundance data for the specified classes
    
#     if cmap is None:
#         cmap = plt.get_cmap("seismic")
#     if cbar_kws is None:
#         cbar_kws = {'label': 'Abundance (log$_2$ AU)'}

#     # get data
#     adata = pdata.prot
#     data = adata.layers[layer] if layer != "X" else adata.X
#     data = data.toarray() if sparse.issparse(data) else data.copy()


#     # log-transform the data
#     abundance_data_log10 = np.log10(data + 1)

#     mid_norm = mcolors.TwoSlopeNorm(vmin=norm_values[0], vcenter=norm_values[1], vmax=norm_values[2])
#     ax = sns.heatmap(abundance_data_log10, yticklabels=True, square=square, annot=annotate, linewidth=linewidth, cmap=cmap, norm=mid_norm, cbar_kws=cbar_kws)

#     return ax

def plot_volcano(ax, pdata=None, classes=None, values=None, method='ttest', fold_change_mode='mean', label=5,
                 label_type='Gene', color=None, alpha=0.5, pval=0.05, log2fc=1, linewidth=0.5,
                 fontsize=8, no_marks=False, de_data=None, return_df=False, **kwargs):
    """
    Plot a volcano plot of differential expression results.

    This function calculates differential expression (DE) between two groups
    and visualizes results as a volcano plot. Alternatively, it can use
    pre-computed DE results (e.g. from `pdata.de()`).

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData, optional): Input pAnnData object. Required if `de_data`
            is not provided.
        classes (str, optional): Sample class column to use for group comparison.
        values (list or dict, optional): Values to compare between groups.
            
            - Legacy list format: `["group1", "group2"]`
            
            - Dictionary format: list of dicts specifying multiple conditions,
              e.g. `[{"cellline": "HCT116", "treatment": "DMSO"},
                     {"cellline": "HCT116", "treatment": "DrugX"}]`.

        method (str): Statistical test method. Default is `"ttest"`. Options are `"ttest"`, `"mannwhitneyu"`, `"wilcoxon"`.
        fold_change_mode (str): Method for computing fold change.
            
            - `"mean"`: log2(mean(group1) / mean(group2))
            
            - `"pairwise_median"`: median of all pairwise log2 ratios.

        label (int, list, or None): Features to highlight.
            
            - If int: label top and bottom *n* features.
            
            - If list of str: label only the specified features.
            
            - If list of two ints: `[top, bottom]` to label asymmetric counts.
            
            - If None: no labels plotted.

        label_type (str): Label content type. Currently `"Gene"` is recommended.
        color (dict, optional): Dictionary mapping significance categories
            to colors. Defaults to grey/red/blue.
        alpha (float): Point transparency. Default is 0.5.
        pval (float): P-value threshold for significance. Default is 0.05.
        log2fc (float): Log2 fold change threshold for significance. Default is 1.
        linewidth (float): Line width for threshold lines. Default is 0.5.
        fontsize (int): Font size for feature labels. Default is 8.
        no_marks (bool): If True, suppress coloring of significant points and
            plot all points in grey. Default is False.
        de_data (pandas.DataFrame, optional): Pre-computed DE results. Must contain
            `"log2fc"`, `"p_value"`, and `"significance"` columns.
        return_df (bool): If True, return both the axis and the DataFrame used
            for plotting. Default is False.
        **kwargs: Additional keyword arguments passed to `matplotlib.pyplot.scatter`.

    Returns:
        ax (matplotlib.axes.Axes): Axis with the volcano plot if `return_df=False`.
        tuple (matplotlib.axes.Axes, pandas.DataFrame): Returned if `return_df=True`.

    Usage Tips:
        mark_volcano: Highlight specific features on an existing volcano plot.  
        - For selective highlighting, set `no_marks=True` to render all points
          in grey, then call `mark_volcano()` to add specific features of interest.

        add_volcano_legend: Add standard legend handles for volcano plots.
        - Use the helper function `add_volcano_legend(ax)` to add standard
          significance legend handles.

    Example:
        Dictionary-style input:
            ```python
            values = [
                {"cellline": "HCT116", "treatment": "DMSO"},
                {"cellline": "HCT116", "treatment": "DrugX"}
            ]
            colors = sns.color_palette("Paired")[4:6]
            color_dict = dict(zip(['downregulated', 'upregulated'], colors))
            ax, df = plot_volcano(ax, pdata, classes="cellline", values=values)
            ```
        Legacy input:
            ```python
            ax, df = plot_volcano(ax, pdata, classes="cellline", values=["A", "B"], color=color_dict)
            add_volcano_legend(ax)
            ```

    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    import matplotlib.patheffects as PathEffects

    if de_data is None and pdata is None:
        raise ValueError("Either de_data or pdata must be provided.")

    if de_data is not None:
        volcano_df = de_data.copy()
    else:
        if values is None:
          raise ValueError("If pdata is provided, values must also be provided.")
        if isinstance(values, list) and isinstance(values[0], dict):
          volcano_df = pdata.de(values=values, method=method, pval=pval, log2fc=log2fc, fold_change_mode=fold_change_mode)
        else:
            volcano_df = pdata.de(class_type=classes, values=values, method=method, pval=pval, log2fc=log2fc, fold_change_mode=fold_change_mode)

    volcano_df = volcano_df.dropna(subset=['p_value']).copy()

    default_color = {'not significant': 'grey', 'upregulated': 'red', 'downregulated': 'blue'}
    if color:
        default_color.update(color)
    elif no_marks:
        default_color = {k: 'grey' for k in default_color}

    scatter_kwargs = dict(s=20, edgecolors='none')
    scatter_kwargs.update(kwargs)
    colors = volcano_df['significance'].astype(str).map(default_color)

    ax.scatter(volcano_df['log2fc'], volcano_df['-log10(p_value)'],
               c=colors, alpha=alpha, **scatter_kwargs)

    ax.axhline(-np.log10(pval), color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(log2fc, color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(-log2fc, color='black', linestyle='--', linewidth=linewidth)

    ax.set_xlabel('$log_{2}$ fold change')
    ax.set_ylabel('-$log_{10}$ p value')

    log2fc_clean = volcano_df['log2fc'].replace([np.inf, -np.inf], np.nan).dropna()
    if log2fc_clean.empty:
        max_abs_log2fc = 1  # default range if nothing valid
    else:
        max_abs_log2fc = log2fc_clean.abs().max() + 0.5
    ax.set_xlim(-max_abs_log2fc, max_abs_log2fc)


    if not no_marks and label not in [None, 0, [0, 0]]:
        if isinstance(label, int):
            upregulated = volcano_df[volcano_df['significance'] == 'upregulated'].sort_values('significance_score', ascending=False)
            downregulated = volcano_df[volcano_df['significance'] == 'downregulated'].sort_values('significance_score', ascending=True)
            label_df = pd.concat([upregulated.head(label), downregulated.head(label)])
        elif isinstance(label, list):
            if len(label) == 2 and all(isinstance(i, int) for i in label):
                upregulated = volcano_df[volcano_df['significance'] == 'upregulated'].sort_values('significance_score', ascending=False)
                downregulated = volcano_df[volcano_df['significance'] == 'downregulated'].sort_values('significance_score', ascending=True)
                label_df = pd.concat([upregulated.head(label[0]), downregulated.head(label[1])])
            else:
                label_lower = [str(l).lower() for l in label]
                label_df = volcano_df[
                volcano_df.index.str.lower().isin(label_lower) |
                volcano_df['Genes'].str.lower().isin(label_lower)
            ]

        else:
            raise ValueError("label must be int or list")

        texts = []
        for i in range(len(label_df)):
            gene = label_df.iloc[i].get('Genes', label_df.index[i])
            txt = plt.text(label_df.iloc[i]['log2fc'],
                           label_df.iloc[i]['-log10(p_value)'],
                           s=gene,
                           fontsize=fontsize,
                           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.6))
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            texts.append(txt)

        adjust_text(texts, expand=(2, 2), arrowprops=dict(arrowstyle='->', color='k', zorder=5))

    # Add group names and DE counts to plot
    def format_group(values_entry, classes):
        if isinstance(values_entry, dict):
            return "/".join(str(v) for v in values_entry.values())
        elif isinstance(values_entry, list) and isinstance(classes, list) and len(values_entry) == len(classes):
            return "/".join(str(v) for v in values_entry)
        return str(values_entry)

    group1 = group2 = ""
    if isinstance(values, list) and len(values) == 2:
        group1 = format_group(values[0], classes)
        group2 = format_group(values[1], classes)

    up_count = (volcano_df['significance'] == 'upregulated').sum()
    down_count = (volcano_df['significance'] == 'downregulated').sum()

    bbox_style = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black')
    
    ax.annotate(group1, xy=(0.98, 1.07), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=fontsize, weight='bold', bbox=bbox_style)
    ax.annotate(f'n={up_count}', xy=(0.98, 1.015), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=fontsize, color=default_color.get('upregulated', 'red'))

    ax.annotate(group2, xy=(0.02, 1.07), xycoords='axes fraction',
                ha='left', va='bottom', fontsize=fontsize, weight='bold', bbox=bbox_style)
    ax.annotate(f'n={down_count}', xy=(0.02, 1.015), xycoords='axes fraction',
                ha='left', va='bottom', fontsize=fontsize, color=default_color.get('downregulated', 'blue'))

    if return_df:
        return ax, volcano_df
    else:
        return ax

def add_volcano_legend(ax, colors=None):
    """
    Add a standard legend for volcano plots.

    This function appends a legend to a volcano plot axis, showing handles for
    upregulated, downregulated, and non-significant features. Colors can be
    customized, but default to grey, red, and blue.

    Args:
        ax (matplotlib.axes.Axes): Axis object to which the legend will be added.

        colors (dict, optional): Custom colors for significance categories.
            Keys must include `"upregulated"`, `"downregulated"`, and
            `"not significant"`. Defaults to:
            
            ```python
            {
                "not significant": "grey",
                "upregulated": "red",
                "downregulated": "blue"
            }
            ```

    Returns:
        None
    """
    from matplotlib.lines import Line2D
    import numpy as np

    default_colors = {'not significant': 'grey', 'upregulated': 'red', 'downregulated': 'blue'}
    if colors is None:
        colors = default_colors.copy()
    else:
        default_colors.update(colors)
        colors = default_colors

    handles = [
        Line2D([0], [0], marker='o', color='w', label='Up', markerfacecolor=colors['upregulated'], markersize=6),
        Line2D([0], [0], marker='o', color='w', label='Down', markerfacecolor=colors['downregulated'], markersize=6),
        Line2D([0], [0], marker='o', color='w', label='NS', markerfacecolor=colors['not significant'], markersize=6)
    ]
    ax.legend(handles=handles, loc='upper right', frameon=True, fontsize=7)

def mark_volcano(ax, volcano_df, label, label_color="black", label_type='Gene', s=10, alpha=1, show_names=True, fontsize=8):
    """
    Mark a volcano plot with specific proteins or genes.

    This function highlights selected features on an existing volcano plot,
    optionally labeling them with names.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        volcano_df (pandas.DataFrame): DataFrame returned by `plot_volcano()` or
            `pdata.de()`, containing differential expression results.
        label (list): Features to highlight. Can also be a nested list, with
            separate lists of features for different cases.
        label_color (str or list, optional): Marker color(s). Defaults to `"black"`.
            If a list is provided, each case receives a different color.
        label_type (str): Type of label to display. Default is `"Gene"`.
        s (float): Marker size. Default is 10.
        alpha (float): Marker transparency. Default is 1.
        show_names (bool): Whether to show labels for the selected features.
            Default is True.
        fontsize (int): Font size for labels. Default is 8.

    Returns:
        ax (matplotlib.axes.Axes): Axis with the highlighted volcano plot.

    Example:
        Highlight specific features on a volcano plot:
            ```python
            fig, ax = plt.subplots()
            ax, df = scplt.plot_volcano(ax, pdata, classes="treatment", values=["ctrl", "drug"])
            ax = scplt.mark_volcano(
                ax, df, label=["P11247", "O35639", "F6ZDS4"],
                label_color="red", s=10, alpha=1, show_names=True
            )
            ```

    Note:
        This function works especially well in combination with
        `plot_volcano(..., no_marks=True)` to render all points in grey,
        followed by `mark_volcano()` to selectively highlight features of interest.
    """

    if not isinstance(label[0], list):
        label = [label]
        label_color = [label_color] if isinstance(label_color, str) else label_color

    for i, label_group in enumerate(label):
        color = label_color[i % len(label_color)] if isinstance(label_color, list) else label_color

        # Match by index or 'Genes' column
        match_df = volcano_df[
            volcano_df.index.isin(label_group) |
            volcano_df['Genes'].isin(label_group)
        ]

        ax.scatter(match_df['log2fc'], match_df['-log10(p_value)'],
                   c=color, s=s, alpha=alpha, edgecolors='none')

        if show_names:
            texts = []
            for _, row in match_df.iterrows():
                text = row['Genes'] if label_type == 'Gene' and pd.notna(row.get('Genes')) else row.name
                txt = ax.text(row['log2fc'], row['-log10(p_value)'],
                              s=text,
                              fontsize=fontsize,
                              color=color,
                              bbox=dict(facecolor='white', edgecolor=color, boxstyle='round'))
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
                texts.append(txt)
            adjust_text(texts, expand=(2, 2),
                        arrowprops=dict(arrowstyle='->', color=color, zorder=5))

    return ax

def plot_rankquant(ax, pdata, classes = None, layer = "X", on = 'protein', cmap=['Blues'], color=['blue'], order = None, s=20, alpha=0.2, calpha=1, exp_alpha = 70, debug = False):
    """
    Plot rank abundance distributions across samples or groups.

    This function visualizes rank abundance of proteins or peptides, optionally
    grouped by sample-level classes. Distributions are drawn as scatter plots
    with adjustable opacity and color schemes. Mean, standard deviation, and
    rank statistics are written to `.var` for downstream annotation.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData): Input pAnnData object.
        classes (str or list of str, optional): One or more `.obs` columns to
            group samples. If None, samples are combined into identifier classes.
        layer (str): Data layer to use. Default is `"X"`.
        on (str): Data level to plot, either `"protein"` or `"peptide"`. Default is `"protein"`.
        cmap (str or list of str): Colormap(s) used for scatter distributions.
            Default is `["Blues"]`.
        color (list of str): List of colors used for scatter distributions.
            Defaults to `["blue"]`.
        order (list of str, optional): Custom order of class categories. If None,
            categories appear in data order.
        s (float): Marker size. Default is 20.
        alpha (float): Marker transparency for distributions. Default is 0.2.
        calpha (float): Marker transparency for class means. Default is 1.
        exp_alpha (float): Exponent for scaling probability density values by
            average abundance. Default is 70.
        debug (bool): If True, print debug information during computation.

    Returns:
        ax (matplotlib.axes.Axes): Axis containing the rank abundance plot.
    
    Example:
        Plot rank abundance grouped by sample size:
            ```python
            import seaborn as sns
            colors = sns.color_palette("Blues", 4)
            cmaps = ["Blues", "Reds", "Greens", "Oranges"]
            order = ["sc", "5k", "10k", "20k"]
            fig, ax = plt.subplots(figsize=(4, 3))
            ax = scplt.plot_rankquant(
                ax, pdata_filter, classes="size",
                order=order,
                cmap=cmaps, color=colors, calpha=1, alpha=0.005
            )
            ```

        Format the plot better:
            ```python
            plt.ylabel("Abundance")
            ax.set_ylim(10**ylims[0], 10**ylims[1])
            legend_patches = [
                mpatches.Patch(color=color, label=label)
                for color, label in zip(colors, order)
            ]
            plt.legend(
                handles=legend_patches, bbox_to_anchor=(0.75, 1),
                loc=2, borderaxespad=0., frameon=False
            )
            ```

        Highlight specific points on the rank-quant plot:
            ```python
            scplt.mark_rankquant(
                ax, pdata_filter, mark_df=prot_sc_df,
                class_values=["sc"], show_label=True,
                color="darkorange", label_type="gene"
            )
            ```

    See Also:
        mark_rankquant: Highlight specific proteins or genes on a rank abundance plot.            
            
    """
    # all the plot_dfs should now be stored in pdata.var
    pdata.rank(classes, on, layer)

    adata = utils.get_adata(pdata, on)
    classes_list = utils.get_classlist(adata, classes = classes, order = order)

    # Ensure colormap and color list match number of classes
    cmap = cmap if cmap and len(cmap) == len(classes_list) else get_color('cmap', n=len(classes_list))
    color = color if color and len(color) == len(classes_list) else get_color('colors', n=len(classes_list))

    for j, class_value in enumerate(classes_list):
        if classes is None or isinstance(classes, (str, list)):
            values = class_value.split('_') if classes is not str else class_value
            rank_data = utils.filter(adata, classes, values, debug=False)

        plot_df = rank_data.to_df().transpose()
        plot_df['Average: '+class_value] = np.nanmean(rank_data.X.toarray(), axis=0)
        plot_df['Stdev: '+class_value] = np.nanstd(rank_data.X.toarray(), axis=0)
        plot_df.sort_values(by=['Average: '+class_value], ascending=False, inplace=True)
        plot_df['Rank: '+class_value] = np.where(plot_df['Average: '+class_value].isna(), np.nan, np.arange(1, len(plot_df) + 1))

        sorted_indices = plot_df.index
        plot_df = plot_df.loc[adata.var.index]
        adata.var['Average: ' + class_value] = plot_df['Average: ' + class_value]
        adata.var['Stdev: ' + class_value] = plot_df['Stdev: ' + class_value]
        adata.var['Rank: ' + class_value] = plot_df['Rank: ' + class_value]
        plot_df = plot_df.reindex(sorted_indices)

        # if taking from pdata.var, can continue from here
        # problem is that we need rank_data, the data consisting of samples from this class to make
        # stats df should have 3 column, average stdev and rank
        # plot_df should only have the abundance 
        stats_df = plot_df.filter(regex = 'Average: |Stdev: |Rank: ', axis=1)
        plot_df = plot_df.drop(stats_df.columns, axis=1)
        print(stats_df.shape) if debug else None
        print(plot_df.shape) if debug else None

        nsample = plot_df.shape[1]
        nprot = plot_df.shape[0]

        # Abundance matrix: shape (nprot, nsample)
        X_matrix = plot_df.values  # shape: (nprot, nsample)
        ranks = stats_df['Rank: ' + class_value].values  # shape: (nprot,)
        mu = np.log10(np.clip(stats_df['Average: ' + class_value].values, 1e-6, None))
        std = np.log10(np.clip(stats_df['Stdev: ' + class_value].values, 1e-6, None))
        # Flatten abundance data (X) and repeat ranks (Y)
        X = X_matrix.flatten(order='F')  # Fortran order stacks column-wise, matching your loop
        Y = np.tile(ranks, nsample)
        # Compute Z-values
        logX = np.log10(np.clip(X, 1e-6, None))
        z = ((logX - np.tile(mu, nsample)) / np.tile(std, nsample)) ** 2
        Z = np.exp(-z * exp_alpha)
        # Remove NaNs
        mask = ~np.isnan(X)
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        print(f'nsample: {nsample}, nprot: {np.max(Y)}') if debug else None

        ax.scatter(Y, X, c=Z, marker='.',cmap=cmap[j], s=s,alpha=alpha)
        ax.scatter(stats_df['Rank: '+class_value], 
                   stats_df['Average: '+class_value], 
                   marker='.', 
                   color=color[j], 
                   alpha=calpha,
                   label=class_value)
        ax.set_yscale('log')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Abundance')

    # format the argument string classes to be first letter capitalized
    legend_title = (
        "/".join(cls.capitalize() for cls in classes)
        if isinstance(classes, list)
        else classes.capitalize() if isinstance(classes, str)
        else None)

    ax.legend(title=legend_title, loc='best', frameon=True, fontsize='small')
    return ax

def mark_rankquant(plot, pdata, mark_df, class_values, layer = "X", on = 'protein', color='red', s=10,alpha=1, show_label=True, label_type='accession'):
    """
    Highlight specific features on a rank abundance plot.

    This function marks selected proteins or peptides on an existing rank
    abundance plot, optionally adding labels. It uses statistics stored in
    `.var` during `plot_rankquant()`.

    Args:
        plot (matplotlib.axes.Axes): Axis containing the rank abundance plot.
        pdata (pAnnData): Input pAnnData object.
        mark_df (pandas.DataFrame): Features to highlight.
            
            - DataFrame: Must include an `"Entry"` and `"accession"` column, and optionally
              `"Gene Names"` if `label_type="gene"`.  
              A typical way to generate this is using
              `scutils.get_upset_query()`, e.g.:

              ```python
              size_upset = scutils.get_upset_contents(pdata_filter, classes="size")
              prot_sc_df = scutils.get_upset_query(size_upset, present=["sc"], absent=["5k", "10k", "20k"])
              ```
            
        class_values (list of str): Class values to highlight (must match those
            used in `plot_rankquant`).
        layer (str): Data layer to use. Default is `"X"`.
        on (str): Data level, either `"protein"` or `"peptide"`. Default is `"protein"`.
        color (str): Marker color. Default is `"red"`.
        s (float): Marker size. Default is 10.
        alpha (float): Marker transparency. Default is 1.
        show_label (bool): Whether to display labels for highlighted features.
            Default is True.
        label_type (str): Label type. Options:
            
            - `"accession"`: show accession IDs.
            - `"gene"`: map to gene names using `"Gene Names"` in `mark_df`.

    Returns:
        ax (matplotlib.axes.Axes): Axis with highlighted features.

    !!! tip 
    
        Works best when paired with `plot_rankquant()`, which stores `Average`,
        `Stdev`, and `Rank` statistics in `.var`. Call `plot_rankquant()` first
        to generate these values, then use `mark_rankquant()` to overlay
        highlights.

    Example:
        Plot rank abundance and highlight specific proteins:
            ```python
            fig, ax = plt.subplots()
            ax = scplt.plot_rankquant(
                ax, pdata_filter, classes="size", order=order,
                cmap=cmaps, color=colors, s=10, calpha=1, alpha=0.005
            )
            size_upset = scutils.get_upset_contents(pdata_filter, classes="size")
            prot_sc_df = scutils.get_upset_query(
                size_upset, present=["sc"], absent=["5k", "10k", "20k"]
            )
            scplt.mark_rankquant(
                ax, pdata_filter, mark_df=prot_sc_df,
                class_values=["sc"], show_label=True,
                color="darkorange", label_type="gene"
            )
            ```python

    See Also:
        plot_rankquant: Generate rank abundance plots with statistics stored in `.var`.
        get_upset_query: Create a DataFrame of proteins based on set intersections (obs membership).
    """
    adata = utils.get_adata(pdata, on)
    names = mark_df['Entry'].tolist()
    
    # TEST: check if names are in the data
    pdata._check_rankcol(on, class_values)

    for j, class_value in enumerate(class_values):
        print('Class: ', class_value)
        
        for i, txt in enumerate(names):
            try:
                x = adata.var['Average: '+class_value].loc[txt]
                y = adata.var['Rank: '+class_value].loc[txt]
            except Exception as e:
                print(f"Name {txt} not found in {on}.var. Check {on} name for spelling errors and whether it is in data.")
                continue
            if show_label:
                if label_type == 'accession':
                    pass
                elif label_type == 'gene':
                    txt = mark_df.loc[mark_df['Entry'] == txt, 'Gene Names'].values[0]
                # elif name_type == 'name':

                plot.annotate(txt, (y,x), xytext=(y+10,x*1.1), fontsize=8)
            plot.scatter(y,x,marker='o',color=color,s=s, alpha=alpha)
    return plot

def plot_venn(ax, pdata, classes, set_colors = 'default', return_contents = False, label_order=None, **kwargs):
    """
    Plot a Venn diagram of shared proteins or peptides across groups.

    This function generates a 2- or 3-set Venn diagram based on presence/absence
    data across specified sample-level classes. For more than 3 sets, use
    `plot_upset()` instead.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData): Input pAnnData object.
        classes (str or list of str): Sample-level classes to partition proteins
            or peptides into sets.
        set_colors (str or list of str): Colors for the sets.
            
            - `"default"`: use internal color palette.
            - list of str: custom color list with length equal to the number of sets.

        return_contents (bool): If True, return both the axis and the underlying
            set contents used for plotting.
        label_order (list of str, optional): Custom order of set labels. Must
            contain the same elements as `classes`.
        **kwargs: Additional keyword arguments passed to matplotlib-venn functions.

    Returns:
        ax (matplotlib.axes.Axes): Axis containing the Venn diagram. Returned if `return_contents=False`

        tuple (matplotlib.axes.Axes, dict): Returned if `return_contents=True`.
        The dictionary maps class labels to sets of feature identifiers.

    Raises:
        ValueError: If number of sets is not 2 or 3.
        ValueError: If `label_order` does not contain the same elements as `classes`.
        ValueError: If custom `set_colors` length does not match number of sets.

    Example:
        Plot a 2-set Venn diagram of shared proteins:
            ```python
            fig, ax = plt.subplots()
            scplt.plot_venn(
                ax, pdata_1mo_snpc, classes="sample",
                set_colors=["#1f77b4", "#ff7f0e"]
            )
            ```

    See Also:
        plot_upset: Plot an UpSet diagram for >3 sets.  
        plot_rankquant: Rank-based visualization of protein/peptide distributions.
    """
    upset_contents = utils.get_upset_contents(pdata, classes, upsetForm=False)

    num_keys = len(upset_contents)
    if set_colors == 'default':
        set_colors = get_color('colors', n=num_keys)
    elif len(set_colors) != num_keys:
        raise ValueError("The number of colors provided must match the number of sets.")
    
    if label_order is not None:
        if set(label_order) != set(upset_contents.keys()):
            raise ValueError("`label_order` must contain the same elements as `classes`.")
        set_labels = label_order
        set_list = [set(upset_contents[label]) for label in set_labels]
    else:
        set_labels = list(upset_contents.keys())
        set_list = [set(value) for value in upset_contents.values()]

    venn_functions = {
        2: lambda: (venn2_unweighted(set_list, ax = ax, set_labels=set_labels, set_colors=tuple(set_colors), alpha=0.5, **kwargs),
                    venn2_circles(subsets=(1, 1, 1), ax = ax,  linewidth=1)),
        3: lambda: (venn3_unweighted(set_list, ax = ax, set_labels=set_labels, set_colors=tuple(set_colors), alpha=0.5, **kwargs),
                    venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), ax = ax, linewidth=1))
    }

    if num_keys in venn_functions:
        ax = venn_functions[num_keys]()
    else:
        raise ValueError("Venn diagrams only accept either 2 or 3 sets. For more than 3 sets, use the plot_upset function.")

    if return_contents:
        return ax, upset_contents
    else:
        return ax

def plot_upset(pdata, classes, return_contents = False, **kwargs):
    """
    Plot an UpSet diagram of shared proteins or peptides across groups.

    This function generates an UpSet plot for >2 sets based on presence/absence
    data across specified sample-level classes. Uses the `upsetplot` package
    for visualization.

    Args:
        pdata (pAnnData): Input pAnnData object.

        classes (str or list of str): Sample-level classes to partition proteins
            or peptides into sets.

        return_contents (bool): If True, return both the UpSet object and the
            underlying set contents used for plotting.

        **kwargs: Additional keyword arguments passed to `upsetplot.UpSet`.  
            See the [upsetplot documentation](https://upsetplot.readthedocs.io/en/stable/)  
            for more details. Common arguments include:

            - `sort_categories_by` (str): How to sort categories. Options are
              `"cardinality"`, `"input"`, `"-cardinality"`, or `"-input"`.
            - `min_subset_size` (int): Minimum subset size to display.

    Returns:
        upset (upsetplot.UpSet): The UpSet plot object.

        tuple (upsetplot.UpSet, pandas.DataFrame): Returned if
        `return_contents=True`. The DataFrame contains set membership as a
        multi-index.

    Example:
        Basic usage with set size categories:
            ```python
            upplot, size_upset = scplt.plot_upset(
                pdata_filter, classes="size", sort_categories_by="-input"
            )
            uplot = upplot.plot()
            uplot["intersections"].set_ylabel("Subset size")
            uplot["totals"].set_xlabel("Protein count")
            plt.show()
            ```

        Optional styling of the plot can also be done:
            ```python
            upplot.style_subsets(
                present=["sc"], absent=["2k", "5k", "10k", "20k"],
                edgecolor="black", facecolor="darkorange", linewidth=2, label="sc only"
            )
            upplot.style_subsets(
                absent=["sc"], present=["2k", "5k", "10k", "20k"],
                edgecolor="white", facecolor="#7F7F7F", linewidth=2, label="in all but sc"
            )
            ```

    See Also:
        plot_venn: Plot a Venn diagram for 2 to 3 sets.  
        plot_rankquant: Rank-based visualization of protein/peptide distributions.
    """

    upset_contents = utils.get_upset_contents(pdata, classes = classes)
    upplot = upsetplot.UpSet(upset_contents, subset_size="count", show_counts=True, facecolor = 'black', **kwargs)

    if return_contents:
        return upplot, upset_contents
    else:
        return upplot

def plot_abundance_2D(ax,data,cases,genes='all', cmap='Blues',color=['blue'],s=20,alpha=[0.2,1],calpha=1):
    
    for j in range(len(cases)):
        vars = ['Abundance: '] + cases[j]
        append_string = '_'.join(vars[1:])

        cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]

        # average abundance of proteins across these columns, ignoring NaN values
        data['Average: '+append_string] = data[cols].mean(axis=1, skipna=True)
        data['Stdev: '+append_string] = data[cols].std(axis=1, skipna=True)

        print(append_string)

    case1_name_string = '_'.join(cases[0][:])
    case2_name_string = '_'.join(cases[1][:])
    
    # find the number for the average column  of the 2 cases
    case1_col = data.columns.get_loc('Average: '+case1_name_string)
    case2_col = data.columns.get_loc('Average: '+case2_name_string)

    # ignore rows where the 2 cases are NaN or 0
    data = data.copy()
    data = data[data.iloc[:,case1_col].notnull()]
    data = data[data.iloc[:,case2_col].notnull()]
    data = data[data.iloc[:,case1_col] != 0]
    data = data[data.iloc[:,case2_col] != 0]

    X = data.iloc[:,case1_col].values
    Y = data.iloc[:,case2_col].values

    # make 2D scatter plot of case1 abundance vs case2 abundance
    ax.scatter(X, Y, marker='.',cmap=cmap, s=s,alpha=alpha[0])
    # set both axis to log
    ax.set_xscale('log')
    ax.set_yscale('log')

    if isinstance(genes, list):
        print('highlighting genes')
        # genes is a list of gene names, so let's extract those that match the accession column
        for i in range(len(genes)):
            # if gene is in data['Gene Symbol'], extract the abundance values for that gene
            if genes[i] in data['Gene Symbol'].values:
                X_highlight = data[data['Gene Symbol']==genes[i]].iloc[:,case1_col].values[0]
                Y_highlight = data[data['Gene Symbol']==genes[i]].iloc[:,case2_col].values[0]
                ax.scatter(X_highlight,Y_highlight,marker='.',color=color[0],s=s,alpha=alpha[1])
                # add gene name to plot
                ax.annotate(genes[i], (X_highlight,Y_highlight), xytext=(X_highlight+10,Y_highlight*1.1), fontsize=10)

    else:
        # plot all genes
        for i, txt in enumerate(data['Gene Symbol']):
            # ax.annotate(txt, (X[i],Y[i]), xytext=(X[i]+10,Y[i]*1.1), fontsize=8)
            ax.scatter(X[i],Y[i],marker='o',color=color[0],s=s,alpha=alpha[1])

    # get min and max of both axes
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # add a 1:1 line, make line hash dotted with alpha = 0.3
    ax.plot([1e-1,1e7],[1e-1,1e7], ls='--', color='grey', alpha=0.3)

    # set x and y limits to be the same
    minval = min(xmin, ymin)
    maxval = max(xmax, ymax)

    ax.set_xlim([minval, maxval])
    ax.set_ylim([minval, maxval])

    return ax

def plot_raincloud(ax,pdata,classes = None, layer = 'X', on = 'protein', order = None, color=['blue'],boxcolor='black',linewidth=0.5, debug = False):
    """
    Plot raincloud distributions of protein or peptide abundances.

    This function generates a raincloud plot (violin + boxplot + scatter)
    to visualize abundance distributions across groups. Summary statistics
    (average, standard deviation, rank) are written into `.var` for downstream
    use with `mark_raincloud()`.

    Args:
        ax (matplotlib.axes.Axes): Axis on which to plot.
        pdata (pAnnData): Input pAnnData object.
        classes (str or list of str, optional): One or more `.obs` columns to
            group samples. If None, all samples are combined.
        layer (str): Data layer to use. Default is `"X"`.
        on (str): Data level, either `"protein"` or `"peptide"`. Default is `"protein"`.
        order (list of str, optional): Custom order of class categories. If None,
            categories appear in data order.
        color (list of str): Colors for each class distribution. Default is `["blue"]`.
        boxcolor (str): Color for boxplot outlines. Default is `"black"`.
        linewidth (float): Line width for box/whisker elements. Default is 0.5.
        debug (bool): If True, return both axis and computed data arrays.

    Returns:
        ax (matplotlib.axes.Axes): If `debug=False`: axis with raincloud plot.

        tuple (matplotlib.axes.Axes, list of np.ndarray): If `debug=True`: `(axis, data_X)` where `data_X` are the transformed abundance distributions per group.

    Note:
        Statistics (`Average`, `Stdev`, `Rank`) are stored in `.var` and can be
        used with `mark_raincloud()` to highlight specific features.

    Example:
        Plot raincloud distributions grouped by sample size:
            ```python
            ax = scplt.plot_raincloud(
                ax, pdata_filter, classes="size",
                order=order, color=colors, linewidth=0.5, debug=False
            )
            scplt.mark_raincloud(
                ax, pdata_filter, mark_df=prot_sc_df,
                class_values=["sc"], color="black"
            )
            ```

    See Also:
        mark_raincloud: Highlight specific features on a raincloud plot.  
        plot_rankquant: Alternative distribution visualization using rank abundance.
    """
    adata = utils.get_adata(pdata, on)

    classes_list = utils.get_classlist(adata, classes = classes, order = order)
    data_X = []

    for j, class_value in enumerate(classes_list):
        rank_data = utils.resolve_class_filter(adata, classes, class_value, debug=True)

        plot_df = rank_data.to_df().transpose()
        plot_df['Average: '+class_value] = np.nanmean(rank_data.X.toarray(), axis=0)
        plot_df['Stdev: '+class_value] = np.nanstd(rank_data.X.toarray(), axis=0)
        plot_df.sort_values(by=['Average: '+class_value], ascending=False, inplace=True)
        plot_df['Rank: '+class_value] = np.where(plot_df['Average: '+class_value].isna(), np.nan, np.arange(1, len(plot_df) + 1))

        sorted_indices = plot_df.index

        plot_df = plot_df.loc[adata.var.index]
        adata.var['Average: ' + class_value] = plot_df['Average: ' + class_value]
        adata.var['Stdev: ' + class_value] = plot_df['Stdev: ' + class_value]
        adata.var['Rank: ' + class_value] = plot_df['Rank: ' + class_value]
        plot_df = plot_df.reindex(sorted_indices)

        stats_df = plot_df.filter(regex = 'Average: |Stdev: |Rank: ', axis=1)
        plot_df = plot_df.drop(stats_df.columns, axis=1)

        nsample = plot_df.shape[1]
        nprot = plot_df.shape[0]

        # merge all abundance columns into one column
        X = np.zeros((nsample*nprot))
        for i in range(nsample):
            X[i*nprot:(i+1)*nprot] = plot_df.iloc[:, i].values

        X = X[~np.isnan(X)] # remove NaN values
        X = X[X != 0] # remove 0 values
        X = np.log10(X)

        data_X.append(X)
    
    print('data_X shape: ', len(data_X)) if debug else None

    # boxplot
    bp = ax.boxplot(data_X, positions=np.arange(1,len(classes_list)+1)-0.06, widths=0.1, patch_artist = True,
                    flierprops=dict(marker='o', alpha=0.2, markersize=2, markerfacecolor=boxcolor, markeredgecolor=boxcolor),
                    whiskerprops=dict(color=boxcolor, linestyle='-', linewidth=linewidth),
                    medianprops=dict(color=boxcolor, linewidth=linewidth),
                    boxprops=dict(facecolor='none', color=boxcolor, linewidth=linewidth),
                    capprops=dict(color=boxcolor, linewidth=linewidth))

    # Violinplot
    vp = ax.violinplot(data_X, points=500, vert=True, positions=np.arange(1,len(classes_list)+1)+0.06,
                showmeans=False, showextrema=False, showmedians=False)

    for idx, b in enumerate(vp['bodies']):
        # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 1])
        # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1.06, idx+2.06)
        # Change to the desired color
        b.set_color(color[idx])
    # Scatterplot data
    for idx in range(len(data_X)):
        features = data_X[idx]
        # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=.1, high=.18, size=len(idxs))
        y = out
        ax.scatter(y, features, s=2., c=color[idx], alpha=0.5)

    if debug:
        return ax, data_X
    else:
        return ax

def mark_raincloud(plot,pdata,mark_df,class_values,layer = "X", on = 'protein',lowest_index=0,color='red',s=10,alpha=1):
    """
    Highlight specific features on a raincloud plot.

    This function marks selected proteins or peptides on an existing
    raincloud plot, using summary statistics written to `.var` during
    `plot_raincloud()`.

    Args:
        plot (matplotlib.axes.Axes): Axis containing the raincloud plot.
        pdata (pAnnData): Input pAnnData object.
        mark_df (pandas.DataFrame): DataFrame containing entries to highlight.
            Must include an `"Entry"` column.
        class_values (list of str): Class values to highlight (must match those
            used in `plot_raincloud`).
        layer (str): Data layer to use. Default is `"X"`.
        on (str): Data level, either `"protein"` or `"peptide"`. Default is `"protein"`.
        lowest_index (int): Offset for horizontal positioning. Default is 0.
        color (str): Marker color. Default is `"red"`.
        s (float): Marker size. Default is 10.
        alpha (float): Marker transparency. Default is 1.

    Returns:
        ax (matplotlib.axes.Axes): Axis with highlighted features.

    !!! tip 
    
        Works best when paired with `plot_raincloud()`, which computes and
        stores the required statistics in `.var`.

    Example:
        Highlight specific proteins on a raincloud plot:
            ```python
            ax = scplt.plot_raincloud(
                ax, pdata_filter, classes="size", order=order,
                color=colors, linewidth=0.5
            )
            scplt.mark_raincloud(
                ax, pdata_filter, mark_df=prot_sc_df,
                class_values=["sc"], color="black"
            )
            ```

    See Also:
        plot_raincloud: Generate raincloud plots with distributions per group.  
        plot_rankquant: Alternative distribution visualization using rank abundance.
    """
    adata = utils.get_adata(pdata, on)
    names = mark_df['Entry'].tolist()
    # TEST: check if names are in the data
    pdata._check_rankcol(on, class_values)

    for j, class_value in enumerate(class_values):
        print('Class: ', class_value)

        for i, txt in enumerate(names):
            try:
                y = np.log10(adata.var['Average: '+class_value].loc[txt])
                x = lowest_index + j + .14 + 0.8
            except Exception as e:
                print(f"Name {txt} not found in {on}.var. Check {on} name for spelling errors and whether it is in data.")
                continue
            plot.scatter(x,y,marker='o',color=color,s=s, alpha=alpha)
    return plot
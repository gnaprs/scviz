"""
This module contains functions for plotting protein data.

Functions:
    plot_significance: Plot significance bars on a given axis.
    plot_cv: Generate a violin plot for the coefficient of variation (CV) of different cases.
    plot_abundance: Plot the abundance of proteins across different cases.
    plot_pca: Plot a PCA of the protein data.
    plot_umap: Plot a UMAP of the protein data.
    plot_pca_scree: Plot a scree plot of the PCA.
    plot_heatmap: Plot a heatmap of protein abundance data.
    plot_volcano: Plot a volcano plot of protein data.
    plot_rankquant: Plot a rank-quantile plot of protein data.
    plot_abundance_2D: Plot the abundance of proteins across different cases in 2D.
    mark_rankquant: Mark the rank-quantile plot with specific proteins.
    plot_raincloud: Plot a raincloud plot of protein data.
    mark_raincloud: Mark the raincloud plot with specific proteins.
    ... more to come

Todo:
    * For future implementation.
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

def get_color(resource_type, n=None):
    """
    Generate a list of colors, a colormap, or a palette from package defaults.

    Parameters:
    - resource_type (str): The type of resource to generate. Options are 'colors', 'cmap', and 'palette'. If 'show', displays all 7 colors.
    - n (int, optional): The number of colors or colormaps to generate. Required for 'colors' and 'cmap'.

    Returns:
    - list of str: If resource_type is 'colors', a list of hex color strings. Repeats colors if n > 7.
    - list of matplotlib.colors.LinearSegmentedColormap: If resource_type is 'cmap'
    - seaborn.color_palette: If resource_type is 'palette'
    - None: If resource_type is 'show', displays the colors and colormaps.

    Example:
    >>> colors = get_color('colors', 5)
    >>> cmap = get_color('cmap')
    >>> palette = get_color('palette')
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
            raise ValueError("Parameter 'n' must be specified when resource_type is 'cmap'")
        if n > len(base_colors):
            warnings.warn(f"Requested {n} colormaps, but only {len(base_colors)} base colors. Reusing from the start.")
        cmaps = []
        for i in range(n):
            color = base_colors[i % len(base_colors)]
            cmap = mcolors.LinearSegmentedColormap.from_list(f'cmap_{i}', ['white', color])
            cmaps.append(cmap)
        return cmaps
    
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
    Plot significance bars on a given axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot the significance bars.
    y (float): The y-coordinate of the bars.
    h (float): The height of the bars.
    x1 (float): The x-coordinate of the first bar.
    x2 (float): The x-coordinate of the second bar.
    col (str): The color of the bars.
    pval (float or str): The p-value used to determine the significance level of the bars.
                         If a float, it is compared against predefined thresholds to determine the significance level.
                         If a string, it is directly used as the significance level.
    fontsize (int): The fontsize of the significance level text.

    Returns:
    None
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

def plot_cv(ax, pdata, classes=None, layer = 'X', on = 'protein', order = None, return_df = False, **kwargs):
    """
    Generate a box and whisker plot for the coefficient of variation (CV) of different cases.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot.
    data (pandas.DataFrame): The data to plot. It should contain columns for each case, with each column containing the CV values for that case.
    cases (list of list of str): A list of cases to plot. Each case is a list of strings that are used to select the columns from the data.
    color (list of str, optional): A list of colors for the box plots of each case. If not provided, all boxes will be blue.

    Returns:
    matplotlib.axes.Axes: The axis with the plotted data.

    Example:
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

def plot_summary(ax, pdata, value='protein_count', classes=None, plot_mean = True, **kwargs):
    if pdata.summary is None:
        pdata._update_summary()

    summary_data = pdata.summary.copy()

    if plot_mean:
        if classes is None:
            raise ValueError("Classes must be specified when plot_mean is True.")
        elif isinstance(classes, str):
            sns.barplot(x=classes, y=value, hue=classes, data=summary_data, ci='sd', ax=ax, **kwargs)
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

def plot_abundance(ax, pdata, namelist=None, layer='X', on='protein',
                   classes=None, return_df=False, order=None, palette=None,
                   log=True, facet=None, height=4, aspect=0.5,
                   plot_points=True, x_label='gene', kind='auto', **kwargs):
    """
    Plot abundance of proteins/peptides using violin + box (inner="box") + strip.

    Parameters:
    ax (matplotlib.axes.Axes): Axis to plot on (ignored if facet is used).
    pdata (pAnnData): Your pAnnData object.
    namelist (list of str): Accessions or gene names to plot.
    layer (str): Data layer name.
    on (str): 'protein' or 'peptide'.
    classes (str or list): obs column(s) to group by (used for color).
    return_df (bool): If True, return DataFrame with replicate + summary values.
    order (list): Custom order of classes.
    palette (list or dict): Color palette.
    log (bool): Plot log2(abundance).
    facet (str or None): obs column to facet by.
    height, aspect (float): For facet layout.
    plot_points (bool): Show stripplot of individual samples.
    x_label (str): Label x-axis as 'gene' or 'accession'.
    kind (str): 'auto' (default), 'violin', or 'bar'. If 'auto', switches to barplot if all groups ≤ 3 samples.
    **kwargs: Extra args passed to violinplot or barplot depending on kind.

    Returns:
    matplotlib.Axes or sns.FacetGrid or pd.DataFrame
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import warnings

    # Get abundance DataFrame
    df = utils.get_abundance(
        pdata, namelist=namelist, layer=layer, on=on,
        classes=classes, log=log, x_label=x_label
    )

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
        violin_kwargs = dict(inner="box", linewidth=1, cut=0, alpha=0.5, scale="width")
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
             pca_params=None, force=False,
             show_labels=False, label_column=None,
             add_ellipses=False, ellipse_kwargs=None):
    """
    Plot PCA scatter plot for classes, protein or peptide abundance.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to plot on (must be 3D if plotting 3 PCs).
    - pdata (scviz.pAnnData): The pAnnData object with .prot, .pep, and .summary.
    - classes (str or list of str or None): 
        - None: plot in grey
        - str: an obs column (e.g. 'treatment') or a protein/gene (e.g. 'UBE4B')
        - list of str: combine multiple obs columns (e.g. ['cellline', 'treatment'])
    - layer (str): The layer to extract from adata (default: "X").
    - on (str): 'protein' or 'peptide' (default: 'protein').
    - cmap (str, list, or colormap):
        - 'default': use get_color() scheme
        - list of colors: used for obs classes
        - colormap name or object: used for continuous abundance coloring
    - s (float): Scatter dot size (default: 20).
    - alpha (float): Dot opacity (default: 0.8).
    - plot_pc (list): PCs to plot (e.g. [1,2] or [1,2,3]).
    - pca_params (dict): Params for PCA, passed to sklearn PCA.
    - force (bool): If True, re-calculate PCA even if it already exists.
    - show_labels (bool or list): 
        - False: no labels
        - True: show all sample names
        - list: only label specified sample names (e.g. ['sample1.raw', 'sample2.raw'])
    - label_column (str or None): Optional column in pdata.summary to use as label source.
    - add_ellipses (bool): If True, overlay confidence ellipses per class (2D only).
        Note: Confidence ellipses are calculated from the group covariance matrix and represent
        a 95% confidence region under a bivariate Gaussian assumption.
    - ellipse_kwargs (dict): Optional kwargs to pass to ellipse patch.

    Returns:
    - ax (matplotlib.axes.Axes): The plot axes.
    - pca (sklearn.decomposition.PCA): The fitted PCA object.

    Examples:
    ---------
    >>> plot_pca(ax, pdata)  # plot in grey
    >>> plot_pca(ax, pdata, classes='treatment')  # color by categorical obs
    >>> plot_pca(ax, pdata, classes=['cellline', 'treatment'])  # combined label
    >>> plot_pca(ax, pdata, classes='UBE4B')  # color by protein expression
    >>> plot_pca(ax, pdata, show_labels=True)  # label each sample
    >>> plot_pca(ax, pdata, show_labels=True, label_column='short_name')  # use custom label
    >>> plot_pca(ax, pdata, classes='treatment', add_ellipses=True)  # add default ellipses
    >>> plot_pca(ax, pdata, classes='treatment', add_ellipses=True, ellipse_kwargs={'alpha': 0.1, 'lw': 2})
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Ellipse
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from sklearn.decomposition import PCA

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

    if 'X_pca' in adata.obsm and not force:
        print(f'PCA already exists in {on} data — using existing.')
    else:
        print(f'Running PCA on {on} using layer {layer}')
        pdata.pca(on=on, layer=layer, **pca_param)

    X_pca = adata.obsm['X_pca']
    pca = adata.uns['pca']

    # Get colors
    color_mapped, cmap_resolved, legend_elements = resolve_pca_colors(adata, classes, cmap, layer=layer)

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
        legend_title = "/".join(c.capitalize() for c in classes) if isinstance(classes, list) else classes.capitalize()
        ax.legend(handles=legend_elements, title=legend_title, loc='best', frameon=False)

    return ax, pca

def resolve_pca_colors(adata, classes, cmap, layer="X"):
    """
    Resolve colors for PCA plot based on classes. Helper function for plot_pca.
    Returns:
        - color_mapped: array-like values to use for coloring
        - cmap_resolved: colormap (only for continuous coloring)
        - legend_elements: legend handles (only for categorical coloring)
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
        elif isinstance(cmap, list):
            palette = cmap
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
        elif isinstance(cmap, list):
            palette = cmap
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
                return resolve_pca_colors(adata, match[0], cmap, layer)
        raise ValueError("Invalid classes input. Must be None, a protein in var_names, or an obs column/list.")

    else:
        raise ValueError("Invalid classes input.")

# NOTE: STRING enrichment plots live in enrichment.py, not here.
# This function is re-documented here for discoverability.
def plot_enrichment_svg(*args, **kwargs):
    """
    Plot STRING enrichment results as an SVG figure.

    NOTE:
        This function is implemented in `enrichment.py`, not `plotting.py`.

    See Also:
        scviz.enrichment.plot_enrichment_svg
    """
    from .enrichment import plot_enrichment_svg as actual_plot
    return actual_plot(*args, **kwargs)

# TODO
def plot_umap(ax, pdata, color = None, layer = "X", on = 'protein', cmap='default', s=20, alpha=.8, umap_params={}, text_size = 10, force = False):
    """
    This function plots the Uniform Manifold Approximation and Projection (UMAP) of the protein data.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on.
        data (pandas.DataFrame): The protein data to plot.
        color (str): The column in the data to color by.
        cmap (matplotlib.colors.Colormap, optional): The colormap to use for the plot. Defaults to 'viridis'.
        s (int, optional): The size of the points in the plot. Defaults to 20.
        alpha (float, optional): The transparency of the points in the plot. Defaults to 0.8.
        umap_params (dict, optional): A dictionary of parameters to pass to the UMAP function. 
            Possible keys are 'min_dist', 'n_components', 'metric', and 'random_state'. 
            Defaults to an empty dictionary, in which case the default UMAP parameters are used.

    Returns:
        ax (matplotlib.axes.Axes): The axes with the plot.
        fit_umap (umap.UMAP): The fitted UMAP object.

    Raises:
        AssertionError: If 'n_components' is 3 but the axes is not a 3D projection.
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
            print(f'UMAP already exists in {on} data, using existing UMAP')
        else:
            pdata.umap(on=on, layer=layer, **umap_param)
    else:
        print(f'UMAP calculation forced, re-calculating UMAP')
        pdata.umap(on=on, layer=layer, **umap_param)

    Xt = adata.obsm['X_umap']
    umap = adata.uns['umap']

    y = utils.get_samplenames(adata, color)
    color_dict = {class_type: i for i, class_type in enumerate(set(y))}
    color_mapped = [color_dict[val] for val in y]
    if cmap == 'default':  
        cmap = get_color('cmap')
    else:
        cmap = cm.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=min(color_mapped), vmax=max(color_mapped))

    if umap_param['n_components'] == 1:
        ax.scatter(Xt[:,0], range(len(Xt)), c=color_mapped, cmap=cmap, s=s, alpha=alpha)
        ax.set_xlabel('UMAP 1', fontsize=text_size)
    if umap_param['n_components'] == 2:
        ax.scatter(Xt[:,0], Xt[:,1], c=color_mapped, cmap=cmap, s=s, alpha=alpha)
        ax.set_xlabel('UMAP 1', fontsize=text_size)
        ax.set_ylabel('UMAP 2', fontsize=text_size)
    if umap_param['n_components'] == 3:
        ax.scatter(Xt[:,0], Xt[:,1], Xt[:,2], c=color_mapped, cmap=cmap, s=s, alpha=alpha)
        ax.set_xlabel('UMAP 1', fontsize=text_size)
        ax.set_ylabel('UMAP 2', fontsize=text_size)
        ax.set_zlabel('UMAP 3', fontsize=text_size)

    # legend
    legend_elements = [mpatches.Patch(color=cmap(norm(color_dict[key])), label=key) for key in color_dict]
    ax.legend(handles=legend_elements, title = color, loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=text_size)

    return ax, umap

def plot_pca_scree(ax, pca):
    """
    Plot a scree plot of the PCA.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to plot the scree plot.
    pca (sklearn.decomposition.PCA): The fitted PCA model.

    Returns:
    ax (matplotlib.axes.Axes): The axes with the plotted scree plot.

    Example:
    >>> import matplotlib.pyplot as plt
    >>> import pandas as pd
    >>> import numpy as np
    >>> from scviz import plotting as scplt
    >>> data = pd.read_excel('tests/data.xlsx', sheet_name='Proteins')
    >>> cases = [['head'],['heart'],['tail']]
    >>> fig, ax = plt.subplots(1,1)
    >>> ax, pca = scplt.plot_pca(ax, data, cases, cmap='viridis', s=20, alpha=.8, plot_pc=[1,2])
    >>> ax = scplt.plot_pca_scree(ax, pca)
    """

    PC_values = np.arange(pca.n_components_) + 1
    ax.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    ax.set_title('Scree Plot')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    
    return ax

# double check
def plot_heatmap(ax, heatmap_data, cmap=cm.get_cmap('seismic'), norm_values=[4,5.5,7], linewidth=.5, annotate=True, square=False, cbar_kws = {'label': 'Abundance (AU)'}):
    """
    Plot annotated heatmap of protein abundance data.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to plot the heatmap.
    heatmap_data (pandas.DataFrame): The data to plot.
    cmap (matplotlib.colors.Colormap): The colormap to use for the heatmap.
    norm_values (list): The low, mid, and high values used to set colorbar scale. Can be assymetric.
    linewidth (float): Plot linewidth.
    annotate (bool): Annotate each heatmap entry with numerical value. True by default.
    square (bool): Make heatmap square. False by default.
    cbar_kws (dict): Pass-through keyword arguments for the colorbar. See `matplotlib.figure.Figure.colorbar()` for more information.

    Returns:
    ax (matplotlib.axes.Axes): The axes with the plotted heatmap.
    """
    # if there are any columns that start with 'Matched in', remove them
    heatmap_data = heatmap_data.loc[:,~heatmap_data.columns.str.contains('Matched in')]
    
    abundance_data_log10 = np.log10(heatmap_data)
    mid_norm = mcolors.TwoSlopeNorm(vmin=norm_values[0], vcenter=norm_values[1], vmax=norm_values[2])
    ax = sns.heatmap(abundance_data_log10, yticklabels=True, square=square, annot=annotate, linewidth=linewidth, cmap=cmap, norm=mid_norm, cbar_kws=cbar_kws)

    return ax

def plot_volcano(ax, pdata=None, classes=None, values=None, method='ttest', fold_change_mode='mean', label=5,
                 label_type='Gene', color=None, alpha=0.5, pval=0.05, log2fc=1, linewidth=0.5,
                 fontsize=8, no_marks=False, de_data=None, return_df=False, **kwargs):
    """
    Plot a volcano plot on the given axes. Calculates DE on pdata across the given class_type and values.
    Alternatively, can use pre-calculated DE data (see pdata.de() dataframe for example input).

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to plot.
    pdata (scviz.pAnnData): The input pdata object.
    classes (str): The class type to use for the comparison.
    values (list or dict): The values to compare. Can be legacy list format or new dict format.
    method (str, optional): The method to use for the comparison. Defaults to 'ttest'.
    fold_change_mode : str
        Method for computing fold change. Options:
        - 'mean' : log2(mean(group1) / mean(group2))
        - 'pairwise_median' : median of all pairwise log2 ratios
    label (int or list): The genes to highlight. If an int, the top and bottom n genes are shown. If a list, only those genes are shown. Can also accept list with 2 numbers to show top and bottom n genes [top, bottom]. If none, no labels will be plotted.
    label_type (str, optional): Label type. Currently only 'Gene' is recommended.
    color (dict, optional): A dictionary mapping significance to colors. Defaults to grey/red/blue.
    alpha (float, optional): Scatter dot transparency. Defaults to 0.5.
    pval (float, optional): The p-value threshold for significance. Defaults to 0.05.
    log2fc (float, optional): The log2 fold change threshold for significance. Defaults to 1.
    linewidth (float, optional): The linewidth for the threshold lines. Defaults to 0.5.
    fontsize (int, optional): Fontsize for gene labels. Defaults to 8.
    no_marks (bool, optional): If True, suppress volcano point coloring. All points are grey.
    de_data (pd.DataFrame): Optional pre-computed DE dataframe. Must contain 'log2fc', 'p_value', 'significance'.
    return_df (bool, optional): If True, return the dataframe used for plotting.
    **kwargs: Extra kwargs passed to matplotlib scatter plot.

    Returns:
    matplotlib.axes.Axes or (ax, df)

    Note:
    Use the helper function `add_volcano_legend(ax)` to add standard volcano legend handles.

    Example:
    >>> ax, df = plot_volcano(ax, pdata, classes='cellline', values=['A', 'B'])
    >>> add_volcano_legend(ax)
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

    ax.scatter(volcano_df['log2fc'], volcano_df['-log10(p_value)'],
               c=volcano_df['significance'].map(default_color), alpha=alpha, **scatter_kwargs)

    ax.axhline(-np.log10(pval), color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(log2fc, color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(-log2fc, color='black', linestyle='--', linewidth=linewidth)

    ax.set_xlabel('$log_{2}$ fold change')
    ax.set_ylabel('-$log_{10}$ p value')

    max_abs_log2fc = np.max(np.abs(volcano_df['log2fc'])) + 0.5
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
    Mark the volcano plot with specific proteins.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to plot.
    volcano_df (pandas.DataFrame): volcano_df data returned from get_protein_DE() or plot_volcano().
    label (list): The genes to highlight. Can be list of list of genes to highlight for each case.
    color (str, optional): The color of the markers. Defaults to 'black'. Can be list of colors for each case.
    s (float, optional): The size of the markers. Defaults to 10.
    alpha (float, optional): The transparency of the markers. Defaults to 1.
    show_names (bool, optional): Whether to show the gene names. Defaults to True.

    Returns:
    ax (matplotlib.axes.Axes): The axes with the plot.

    Example:
    >>> fig, ax = plt.subplots(1,1)
    >>> ax, volcano_df = scplt.plot_volcano(ax, data, cases, log2fc=0.5, pval=0.05, alpha=0.5, fontsize=6, label=[1,2,3]);
    >>> ax = scplt.mark_volcano(ax, data, cases, label=['P11247','O35639','F6ZDS4'], color='red', s=10, alpha=1, show_names=True)
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
    Plot rank abundance of proteins across different classes.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot.
    pdata (scviz.pAnnData): The input pdata object.
    classes (list of str): A list of classes to plot. If None, all .obs are combined into identifier classes. Default is None.
    layer (str, optional): The layer to use for the plot. Default is 'X'.
    on (str, optional): The data to use for the plot. Default is 'protein'.
    cmap (str, optional): The colormap to use for the scatter plot. Default is 'Blues'.
    color (list of str, optional): A list of colors for the scatter plots of each class. If not provided, all plots will be blue.
    order (list of str, optional): The order of the classes to plot. If not provided, the classes will be plotted in the order they appear in the data.
    s (float, optional): The marker size. Default is 20.
    alpha (float, optional): The marker transparency. Default is 0.2.
    calpha (float, optional): The marker transparency for distribution dots. Default is 1.
    append_var (bool, optional): If True, append the average and stdev values to the pdata.[on].var. Default is True. Needs to be True for mark_rankquant to work.
    exp_alpha (float, optional): The exponent for the pdf value based on average abundance. Default is 70.
    
    Example:
    >>> colors = sns.color_palette("Blues", 4)
    >>> cmaps = ['Blues', 'Reds', 'Greens', 'Oranges']
    >>> fig, ax = plt.subplots(figsize=(4,3))
    >>> ax = scplt.plot_rankquant(ax, pdata_filter, classes = 'size', order = ['sc', '5k','10k', '20k'], cmap = cmaps, color=colors, calpha = 1, alpha = 0.005)
    
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
    # example of further styling
    # upplot.style_subsets(present=["sc"], absent=['5k','10k','20k'],facecolor="black", label="sc only")
    # upplot.style_subsets(absent=["sc"], present=['5k','10k','20k'],facecolor="red", label="in all but sc")
    # uplot = upplot.plot(fig = fig)

    # uplot["intersections"].set_ylabel("Subset size")
    # uplot["totals"].set_xlabel("Protein count")

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
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

from scviz import utils

sns.set_theme(context='paper', style='ticks')

def get_color(resource_type, n=None):
    """
    Generate a list of colors, a colormap, or a palette from package defaults.

    Parameters:
    - resource_type (str): The type of resource to generate. Options are 'colors', 'cmap', and 'palette'.
    - n (int, optional): The number of colors to generate. Only used if resource_type is 'colors'.

    Returns:
    - list of str or matplotlib.colors.Colormap or seaborn.color_palette: A list of colors, a colormap, or a palette.

    Example:
    >>> colors = get_color_resources('colors', 5)
    >>> cmap = get_color('cmap')
    >>> palette = get_color('palette')
    """

    # --- 
    # Create a list of colors
    colors = ['#FC9744', '#00AEE8', '#9D9D9D', '#6EDC00', '#F4D03F', '#FF0000', '#A454C7']
    cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)
    palette = sns.color_palette(colors)
    # ---

    if resource_type == 'colors':
        if n is None:
            raise ValueError("Parameter 'n' must be specified when resource_type is 'colors'")
        return colors[:n]
    elif resource_type == 'cmap':
        return cmap
    elif resource_type == 'palette':
        return palette
    else:
        raise ValueError("Invalid resource_type. Options are 'colors', 'cmap', and 'palette'")

def plot_significance(ax, x1, x2, y, h, col, pval):
    """
    Plot significance bars on a given axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot the significance bars.
    x1 (float): The x-coordinate of the first bar.
    x2 (float): The x-coordinate of the second bar.
    y (float): The y-coordinate of the bars.
    h (float): The height of the bars.
    col (str): The color of the bars.
    pval (float or str): The p-value used to determine the significance level of the bars.
                         If a float, it is compared against predefined thresholds to determine the significance level.
                         If a string, it is directly used as the significance level.

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
    ax.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color=col)

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

# add function to label file name?
# TODO: implement like pl.umap (sc.pl.umap(pdata.prot, color = ['P62258-1','P12814-1','Q13509', 'type'])) to return list of ax?
# TODO: if protein_group, then use cbar and cmap to color by protein abundance
def plot_pca(ax, pdata, color_by = None, layer = "X", on = 'protein', cmap='default', s=20, alpha=.8, plot_pc=[1,2], pca_params={}, force=False):
    """
    Plot PCA scatter plot.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the scatter plot.
    - pdata (scviz.pAnnData): The input pdata object.
    - color_by (list): List of classes to color by, can be single string or a list of strings.
    - cmap (matplotlib.colors.Colormap, optional): The colormap to use for coloring the scatter plot. Default is 'default', using the scviz plotting default color scheme get_color().
    - s (float, optional): The marker size. Default is 20.
    - alpha (float, optional): The marker transparency. Default is 0.8.
    - plot_pc (list, optional): The principal components to plot. Default is PC1 and PC2, as in [1, 2]. Can also include 3 components to plot in 3D, as in [1,2,3]
    - force (bool, optional): If True, force PCA calculation even if already exists in pdata.{on}. Default is False. Use to force re-calculation on a different layer.

    Returns:
    - ax (matplotlib.axes.Axes): The axes with the scatter plot.
    - pca (sklearn.decomposition.PCA): The fitted PCA model.

    Raises:
    - AssertionError: If the axes is not a 3D projection but plot_pc includes 3 components.

    Example:
    >>> from scviz import pAnnData as pAnnData
    >>> from scviz import plotting as scplt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> pdata = pAnnData.import_proteomeDiscoverer(prot_file='prot.txt', pep_file='pep.txt')
    >>> color = 'type'
    >>> fig, ax = plt.subplots(1,1)
    >>> ax, pca = scplt.plot_pca(ax, pdata, on = 'protein', layer = 'X_impute_median', color = color, s=20, alpha=.8, plot_pc=[1,2])
    """

    if len(plot_pc) == 3:
        assert ax.name == '3d', "The ax must be a 3D projection, please define projection='3d'"

    pc_x=plot_pc[0]-1
    pc_y=plot_pc[1]-1
    if len(plot_pc) == 3:
        pc_z=plot_pc[2]-1

    if on == 'protein':
        adata = pdata.prot
    elif on == 'peptide':
        adata = pdata.pep
    else:
        raise ValueError("Invalid value for 'on'. Options are 'protein' or 'peptide'.")

    default_pca_params = {'n_comps': min(len(adata.obs_names), len(adata.var_names))-1, 'random_state': 42}
    pca_param = {**default_pca_params, **(pca_params if pca_params else {})}

    if 'X_pca' in adata.obsm.keys():
        if force == False:
            print(f'PCA already exists in {on} data, using existing PCA')
        else:
            print(f'PCA already exists in {on} data, force is True, re-calculating PCA')
            print(f'Calculating PCA on {on} data using {layer} layer')
            pdata.pca(on=on, layer=layer, **pca_param)
    else:
        print(f'Calculating PCA on {on} data using {layer} layer')
        pdata.pca(on=on, layer=layer, **pca_param)

    X_pca = adata.obsm['X_pca']
    pca = adata.uns['pca']

    # TODO: Fix
    # GREY COLOR (color is None)
    if color_by == None:
        color_mapped = ['grey' for i in range(len(X_pca)[0])]
        cmap = 'Greys'
        legend_elements = [mpatches.Patch(color='grey', label='All samples') for i in range(len(X_pca)[0])]
        pass

    # TODO: Fix
    # CONTINUOUS COLOR (color is a protein in adata.var_names)
    if color_by in adata.var_names:
        pass

    # CATEGORICAL COLOR (color is a class/subclass in adata.obs.columns)
    if color_by in adata.obs.columns:
        y = utils.get_samplenames(adata, color_by)

        if cmap == 'default':
            unique_classes = len(set(y))
            colors = get_color('colors', n=unique_classes)
            color_dict = {class_type: colors[i] for i, class_type in enumerate(set(y))}
            color_mapped = [color_dict[val] for val in y]
            cmap = None
            legend_elements = [mpatches.Patch(color=color_dict[key], label=key) for i, key in enumerate(color_dict)]
        else:
            color_dict = {class_type: i for i, class_type in enumerate(set(y))}
            color_mapped = [color_dict[val] for val in y]
            cmap = cm.get_cmap(cmap)
            norm = mcolors.Normalize(vmin=min(color_mapped), vmax=max(color_mapped))
            legend_elements = [mpatches.Patch(color=cmap(norm(color_dict[key])), label=key) for key in color_dict]

    # FIX for list of strings (combined color_by)

    if len(plot_pc) == 2:
        ax.scatter(X_pca[:,pc_x], X_pca[:,pc_y], c=color_mapped, cmap=cmap, s=s, alpha=alpha)
        ax.set_xlabel('PC'+str(pc_x+1)+' ('+str(round(pca['variance_ratio'][pc_x]*100,2))+'%)')
        ax.set_ylabel('PC'+str(pc_y+1)+' ('+str(round(pca['variance_ratio'][pc_y]*100,2))+'%)')

    elif len(plot_pc) == 3:
        ax.scatter(X_pca[:,pc_x], X_pca[:,pc_y], X_pca[:, pc_z], c=color_mapped, cmap=cmap, s=s, alpha=alpha)
        ax.set_xlabel('PC'+str(pc_x+1)+' ('+str(round(pca['variance_ratio'][pc_x]*100,2))+'%)')
        ax.set_ylabel('PC'+str(pc_y+1)+' ('+str(round(pca['variance_ratio'][pc_y]*100,2))+'%)')
        ax.set_zlabel('PC'+str(pc_z+1)+' ('+str(round(pca['variance_ratio'][pc_z]*100,2))+'%)')

    # legend
    ax.legend(handles=legend_elements, title = color_by, loc='upper right', bbox_to_anchor=(1.35, 1), frameon=False)

    return ax, pca

# TODO
def plot_umap(ax, pdata, color_by = None, layer = "X", on = 'protein', cmap='default', s=20, alpha=.8, umap_params={}, text_size = 10, force = False):
    """
    This function plots the Uniform Manifold Approximation and Projection (UMAP) of the protein data.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on.
        data (pandas.DataFrame): The protein data to plot.
        color_by (str): The column in the data to color by.
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

    y = utils.get_samplenames(adata, color_by)
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
    ax.legend(handles=legend_elements, title = color_by, loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=text_size)

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

# double check
def plot_volcano(ax, pdata = None, class_type = None, values = None, on = 'protein', method='ttest', label=5, color=None, alpha=0.5, pval=0.05, log2fc=1, linewidth=0.5, fontsize = 8, de_data = None):
    """
    Plot a volcano plot on the given axes. Calculates DE on pdata across the given class_type and values. Alternatively, can use pre-calculated DE data (see pdata.de() dataframe for example input).

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to plot.
    pdata (scviz.pAnnData): The input pdata object.
    class_type (str): The class type to use for the comparison.
    values (list): The values to compare.
    on (str, optional): The data to use for the comparison. Defaults to 'protein'.
    method (str, optional): The method to use for the comparison. Defaults to 'ttest'.
    label (int or list): The genes to highlight. If an int, the top and bottom n genes are shown. If a list, only those genes are shown. Can also accept list with 2 numbers to show top and bottom n genes [top, bottom]. If none, no labels will be plotted.
    color (dict, optional): A dictionary mapping significance to colors. Defaults to {'not significant': 'grey', 'upregulated': 'red', 'downregulated': 'blue'}.
    alpha (float, optional): The alpha value for the scatter plot. Defaults to 0.5.
    pval (float, optional): The p-value threshold for significance. Defaults to 0.05.
    log2fc (float, optional): The log2 fold change threshold for significance. Defaults to 1.
    linewidth (float, optional): The linewidth for the significance lines. Defaults to 0.5.
    fontsize (int, optional): The fontsize for the gene labels. Defaults to 8.

    Returns:
    ax (matplotlib.axes.Axes): The axes with the plot.
    volcano_df (pandas.DataFrame): The data used for the plot, with additional columns for significance.

    Example:
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.lines import Line2D
    >>> from scviz import plotting as scplt
    >>> import pandas as pd
    
    >>> data = pd.read_excel('tests/data.xlsx', sheet_name='Proteins')
    >>> cases = [['head'],['heart'],['tail']]
    >>> fig, ax = plt.subplots(1,1)
    >>> name_list = ['P11247','O35639','F6ZDS4']
    >>> ax, volcano_df = plot_volcano(ax, data, cases, log2fc=0.5, pval=0.05, alpha=0.5, fontsize=6, label=name_list);
    
    >>> # Create custom artists
    >>> upregulated = Line2D([0], [0], marker='o', color='w', label='Upregulated', markerfacecolor='red', markersize=6)
    >>> downregulated = Line2D([0], [0], marker='o', color='w', label='Downregulated', markerfacecolor='blue', markersize=6)
    >>> no_significance = Line2D([0], [0], marker='o', color='w', label='No significance', markerfacecolor='grey', markersize=6)

    >>> # Add the legend
    >>> plt.legend(handles=[upregulated, downregulated, no_significance], loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0., frameon=True, prop={'size': 7})
    >>> plt.show()
    """
    if de_data is None and pdata is None:
        raise ValueError("Either de_data or pdata must be provided.")

    if de_data is not None:
        volcano_df = de_data
    else:
        if class_type is None or values is None:
            raise ValueError("If pdata is provided, class_type and values must also be provided.")
        volcano_df = pdata.de(class_type, values, on = on, method = method, pval = pval, log2fc = log2fc)
    
    volcano_df = volcano_df.dropna(subset=['p_value'])
    
    default_color = {'not significant': 'grey', 'upregulated': 'red', 'downregulated': 'blue'}
    if color:
        default_color.update(color)

    ax.scatter(volcano_df['log2fc'], volcano_df['-log10(p_value)'], c=volcano_df['significance'].map(default_color), alpha=alpha)
    ax.axhline(-np.log10(pval), color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(log2fc, color='black', linestyle='--', linewidth=linewidth)
    ax.axvline(-log2fc, color='black', linestyle='--', linewidth=linewidth)
    ax.set_xlabel('$log_{2}$ fold change')
    ax.set_ylabel('-$log_{10}$ p value')

    max_abs_log2fc = np.max(np.abs(volcano_df['log2fc'])) + 0.5
    ax.set_xlim(-max_abs_log2fc, max_abs_log2fc)

    if label is None or label == 0 or label == [0,0]:
        pass
    else:
        if isinstance(label, int):
            upregulated = volcano_df[volcano_df['significance'] == 'upregulated'].sort_values('significance_score', ascending=False)
            downregulated = volcano_df[volcano_df['significance'] == 'downregulated'].sort_values('significance_score', ascending=True)
            
            n_upregulated = min(label, len(upregulated))
            n_downregulated = min(label, len(downregulated))
        
            label_df = pd.concat((upregulated.head(n_upregulated), downregulated.head(n_downregulated)))
        elif isinstance(label, list):
            if len(label) == 2 and all(isinstance(i, int) for i in label):
                upregulated = volcano_df[volcano_df['significance'] == 'upregulated'].sort_values('significance_score', ascending=False)
                downregulated = volcano_df[volcano_df['significance'] == 'downregulated'].sort_values('significance_score', ascending=True)

                n_upregulated = min(label[0], len(upregulated))
                n_downregulated = min(label[1], len(downregulated))
            
                label_df = pd.concat((upregulated.head(n_upregulated), downregulated.head(n_downregulated)))
            else:
                label_df = volcano_df.loc[volcano_df.index.isin(label)]
        else:
            raise ValueError('label_list must be an int or a list')

        texts = []
        for i in range(len(label_df)):
            txt = plt.text(x = label_df.iloc[i]['log2fc'],
                                y = label_df.iloc[i]['-log10(p_value)'],
                                s = label_df.index[i],
                                fontsize = fontsize,
                                weight = 'bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            texts.append(txt)
        adjust_text(texts, expand=(2, 2), arrowprops = dict(arrowstyle = '->', color = 'k', zorder = 5))

    return ax

# double check
def mark_volcano(ax, volcano_df, label, label_color="black", s=10, alpha=1, show_names=True, fontsize=8):
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
    >>> import matplotlib.pyplot as plt
    >>> from scviz import plotting as scplt
    >>> import pandas as pd
    >>> data = pd.read_excel('tests/data.xlsx', sheet_name='Proteins')
    >>> cases = [['head'],['heart'],['tail']]
    >>> fig, ax = plt.subplots(1,1)
    >>> ax, volcano_df = scplt.plot_volcano(ax, data, cases, log2fc=0.5, pval=0.05, alpha=0.5, fontsize=6, label=[1,2,3]);
    >>> ax = scplt.mark_volcano(ax, data, cases, label=['P11247','O35639','F6ZDS4'], color='red', s=10, alpha=1, show_names=True)
    """

    # if label is a list of list, then we need to plot each list of genes with a different color
    # else if label is a list of genes, then we need to plot all genes with the same color

    # check if label is a list of list or a list of strings
    if isinstance(label[0], list):
        for i in range(len(label)):
            label_df = volcano_df.loc[volcano_df.index.isin(label[i])]
            ax.scatter(label_df['log2fc'], label_df['-log10(p_value)'], c=label_color[i], alpha=alpha)

            if show_names:
                texts = []
                for j in range(len(label_df)):
                    txt = plt.text(x = label_df.iloc[j]['log2fc']+1,
                                        y = label_df.iloc[j]['-log10(p_value)']+1,
                                        s = label_df.index[j],
                                        fontsize = fontsize,
                                        color = label_color[i],
                                        weight = 'bold', bbox=dict(facecolor='white', edgecolor=label_color[i], boxstyle='round'))
                    
                    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
                    texts.append(txt)
                adjust_text(texts, expand=(2, 2), arrowprops = dict(arrowstyle = '->', color = 'k', zorder = 5))
    elif isinstance(label, list):
        label_df = volcano_df.loc[volcano_df.index.isin(label)]
        ax.scatter(label_df['log2fc'], label_df['-log10(p_value)'], c=label_color, alpha=alpha)

        if show_names:
            texts = []
            for i in range(len(label_df)):
                txt = plt.text(x = label_df.iloc[i]['log2fc']+1,
                                    y = label_df.iloc[i]['-log10(p_value)']+1,
                                    s = label_df.index[i],
                                    fontsize = fontsize,
                                    color = label_color,
                                    weight = 'bold', bbox=dict(facecolor='white', edgecolor=label_color, boxstyle='round'))
                
                txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
                texts.append(txt)
            adjust_text(texts, expand=(2, 2), arrowprops = dict(arrowstyle = '->', color = label_color, zorder = 5))

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
    
    adata = utils.get_adata(pdata, on)
    classes_list = utils.get_classlist(adata, classes = classes, order = order)

    for j, class_value in enumerate(classes_list):
        if classes is None:
            values = class_value.split('_')
            # print(f'Classes: {classes}, Values: {values}') if debug else None
            rank_data = utils.filter(adata, classes, values, suppress_warnings=True)
        elif isinstance(classes, str):
            # print(f'Class: {classes}, Value: {class_value}') if debug else None
            rank_data = utils.filter(adata, classes, class_value, suppress_warnings=True)
        elif isinstance(classes, list):
            values = class_value.split('_')
            # print(f'Classes: {classes}, Values: {values}') if debug else None
            rank_data = utils.filter(adata, classes, values, suppress_warnings=True)

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
        Y = np.zeros((nsample*nprot))
        Z = np.zeros((nsample*nprot)) # pdf value based on average abundance
        for i in range(nsample):
            X[i*nprot:(i+1)*nprot] = plot_df.iloc[:, i].values
            Y[i*nprot:(i+1)*nprot] = stats_df['Rank: '+class_value].values
            mu = np.log10(stats_df['Average: '+class_value].values)
            std = np.log10(stats_df['Stdev: '+class_value].values)
            z = (np.log10(X[i*nprot:(i+1)*nprot]) - mu)/std
            z = np.power(z,2)
            Z[i*nprot:(i+1)*nprot] = np.exp(-z*exp_alpha)

        # remove NaN values
        Z = Z[~np.isnan(X)]
        Y = Y[~np.isnan(X)]
        X = X[~np.isnan(X)]

        print(f'nsample: {nsample}, nprot: {np.max(Y)}') if debug else None

        ax.scatter(Y, X, c=Z, marker='.',cmap=cmap[j], s=s,alpha=alpha)
        ax.scatter(stats_df['Rank: '+class_value], stats_df['Average: '+class_value], marker='.', color=color[j], alpha=calpha)
        ax.set_yscale('log')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Abundance')

    return ax

def mark_rankquant(plot, pdata, names, class_values, layer = "X", on = 'protein',color='red',s=10,alpha=1,show_names=True):
    adata = utils.get_adata(pdata, on)
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
            if show_names:
                plot.annotate(txt, (y,x), xytext=(y+10,x*1.1), fontsize=8)
            plot.scatter(y,x,marker='o',color=color,s=s, alpha=alpha)
    return plot

def plot_venn(ax, pdata, classes, **kwargs):
    set_dict = utils.get_upset_contents(pdata, classes, upsetForm=False)

    num_keys = len(set_dict)
    set_colors = get_color('colors', n=num_keys)
    set_labels = list(set_dict.keys())
    set_list = [set(value) for value in set_dict.values()]

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

    return ax

def plot_upset(pdata, classes, **kwargs):
    # example of further styling
    # upplot.style_subsets(present=["sc"], absent=['5k','10k','20k'],facecolor="black", label="sc only")
    # upplot.style_subsets(absent=["sc"], present=['5k','10k','20k'],facecolor="red", label="in all but sc")
    # uplot = upplot.plot(fig = fig)

    # uplot["intersections"].set_ylabel("Subset size")
    # uplot["totals"].set_xlabel("Protein count")

    data_upset = utils.get_upset_contents(pdata, classes = classes)
    upplot = upsetplot.UpSet(data_upset, subset_size="count", show_counts=True, facecolor = 'black', **kwargs)

    return upplot


# TODO: make function work from get_abundance or get_abundance_query, actually plot the protein abundances - refer to graph from tingyu
def plot_abundance(ax, prot_data, classes, genelist, cmap='Blues', color=['blue'], s=20, alpha=0.2, calpha=1):
    """
    Plot the abundance of proteins across different cases.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot.
    prot_data (pandas.DataFrame): The protein data to plot.
    cases (list of list of str): A list of cases to plot. Each case is a list of strings that are used to select the columns from the data.
    cmap (str, optional): The colormap to use for the scatter plot. Default is 'Blues'.
    color (list of str, optional): A list of colors for the scatter plots of each case. If not provided, all plots will be blue.
    s (float, optional): The marker size. Default is 20.
    alpha (float, optional): The marker transparency. Default is 0.2.
    calpha (float, optional): The marker transparency for the case average. Default is 1.

    Returns:
    matplotlib.axes.Axes: The axis with the plotted data.

    Example:
    >>> import matplotlib.pyplot as plt
    >>> import pandas as pd
    >>> cases = [['a','50g'],['b','50g'],['a','30g'],['b','30g']]
    >>> fig, ax = plt.subplots()
    >>> plot_abundance(ax, prot_data, cases, cmap='Blues', color = ['blue','red','green','orange'])
    """

    for j in range(len(cases)):
        vars = ['Abundance: '] + cases[j]
        append_string = '_'.join(vars[1:])

        cols = [col for col in prot_data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]

        # average abundance of proteins across these columns, ignoring NaN values
        prot_data['Average: '+append_string] = prot_data[cols].mean(axis=1, skipna=True)
        prot_data['Stdev: '+append_string] = prot_data[cols].std(axis=1, skipna=True)

        nsample = len(cols)

        # merge all abundance columns into one column
        X = np.zeros((nsample*len(prot_data)))
        Y = np.zeros((nsample*len(prot_data)))
        Z = np.zeros((nsample*len(prot_data))) # pdf value based on average abundance
        for i in range(nsample):
            X[i*len(prot_data):(i+1)*len(prot_data)] = prot_data[cols[i]].values

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
        if classes is None:
            values = class_value.split('_')
            print(f'Classes: {classes}, Values: {values}') if debug else None
            rank_data = utils.filter(adata, classes, values, suppress_warnings=True)
        elif isinstance(classes, str):
            print(f'Class: {classes}, Value: {class_value}') if debug else None
            rank_data = utils.filter(adata, classes, class_value, suppress_warnings=True)
        elif isinstance(classes, list):
            values = class_value.split('_')
            print(f'Classes: {classes}, Values: {values}') if debug else None
            rank_data = utils.filter(adata, classes, values, suppress_warnings=True)

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

        X = X[~np.isnan(X)]
        X = np.log10(X)

        data_X.append(X)
        
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

    return ax

def mark_raincloud(plot,pdata,names,class_values,layer = "X", on = 'protein',lowest_index=0,color='red',s=10,alpha=1,show_names=True):
    adata = utils.get_adata(pdata, on)
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
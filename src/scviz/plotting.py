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
from upsetplot import plot, generate_counts, from_contents, query, UpSet
from adjustText import adjust_text
import umap.umap_ as umap

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

def plot_cv(ax,data,cases,color=['blue']):
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
    >>> import matplotlib.pyplot as plt
    >>> import pandas as pd
    >>> cases = [['a','50g'],['b','50g'],['a','30g'],['b','30g']]
    >>> fig, ax = plt.subplots()
    >>> plot_cv(ax, data, cases, color = ['blue','red','green','orange'])
    """

    for j in range(len(cases)):
        vars = ['CV'] + cases[j]
        cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]

        nsample = len(cols)

        # merge all abundance columns into one column
        X = np.zeros((nsample*len(data)))
        for i in range(nsample):
            X[i*len(data):(i+1)*len(data)] = data[cols[i]].values
        # remove nans
        X = X[~np.isnan(X)]/100
        # make box and whiskers plot of coefficient of variation
        ax.boxplot(X, positions=[j+1], widths=0.5, patch_artist = True,
                    flierprops=dict(marker='o', alpha=0.2, markersize=2, markerfacecolor=color[j], markeredgecolor=color[j]),
                    whiskerprops=dict(color='black', linestyle='-', linewidth=0.5),
                    medianprops=dict(color='black', linewidth=0.5),
                    boxprops=dict(facecolor=color[j], color='black', linewidth=0.5),
                    capprops=dict(color=color[j], linewidth=0.5))
    return ax

# TODO: make function work from get_abundance or get_abundance_query
def plot_abundance(ax, prot_data, cases, cmap='Blues', color=['blue'], s=20, alpha=0.2, calpha=1):
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

def plot_pca(ax, data, cases, cmap=cm.get_cmap('viridis'), s=20, alpha=.8, plot_pc=[1,2]):
    """
    Plot PCA scatter plot.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot the scatter plot.
    - data (numpy.ndarray): The input data array.
    - cases (list): The list of case names.
    - cmap (matplotlib.colors.Colormap, optional): The colormap to use for coloring the scatter plot. Default is 'viridis'.
    - s (float, optional): The marker size. Default is 20.
    - alpha (float, optional): The marker transparency. Default is 0.8.
    - plot_pc (list, optional): The principal components to plot. Default is PC1 and PC2, as in [1, 2]. Can also include 3 components to plot in 3D, as in [1,2,3]

    Returns:
    - ax (matplotlib.axes.Axes): The axes with the scatter plot.
    - pca (sklearn.decomposition.PCA): The fitted PCA model.

    Raises:
    - AssertionError: If the axes is not a 3D projection but plot_pc includes 3 components.

    Example:
    >>> import matplotlib.pyplot as plt
    >>> import pandas as pd
    >>> import numpy as np
    >>> from scviz import plotting as scplt
    >>> data = pd.read_excel('tests/data.xlsx', sheet_name='Proteins')
    >>> cases = [['head'],['heart'],['tail']]
    >>> fig, ax = plt.subplots(1,1)
    >>> ax, pca = scplt.plot_pca(ax, data, cases, cmap='viridis', s=20, alpha=.8, plot_pc=[1,2])
    """
    dict_data = utils.get_abundance(data, cases, abun_type='raw')

    if len(plot_pc) == 3:
        assert ax.name == '3d', "The ax must be a 3D projection, please define projection='3d'"

    pc_x=plot_pc[0]-1
    pc_y=plot_pc[1]-1
    if len(plot_pc) == 3:
        pc_z=plot_pc[2]-1

    # make stack of all abundance data
    X = np.hstack([np.array(dict_data[list(dict_data.keys())[i]]) for i in range(len(dict_data))])
    X = X.T

    # make sample array
    y = np.hstack([np.repeat(i, dict_data[list(dict_data.keys())[i]].shape[1]) for i in range(len(dict_data))])
    print(f'BEFORE: Number of samples|Number of proteins: {X.shape}')

    # remove columns that contain nan in X
    X = X[:, ~np.isnan(X).any(axis=0)]
    print(f'AFTER: Number of samples|Number of proteins: {X.shape}')
    X = np.log2(X+1)

    Xnorm = (X - X.mean(axis=0)) / X.std(axis=0)

    pca = PCA()
    Xt = pca.fit_transform(Xnorm)
  
    if len(plot_pc) == 2:
        ax.scatter(Xt[:,pc_x], Xt[:,pc_y], c=y, cmap=cmap, s=s, alpha=alpha)
        ax.set_xlabel('PC'+str(pc_x+1)+' ('+str(round(pca.explained_variance_ratio_[pc_x]*100,2))+'%)')
        ax.set_ylabel('PC'+str(pc_y+1)+' ('+str(round(pca.explained_variance_ratio_[pc_y]*100,2))+'%)')

    elif len(plot_pc) == 3:
        ax.scatter(Xt[:,pc_x], Xt[:,pc_y], Xt[:, pc_z], c=y, cmap=cmap, s=s, alpha=alpha)
        ax.set_xlabel('PC'+str(pc_x+1)+' ('+str(round(pca.explained_variance_ratio_[pc_x]*100,2))+'%)')
        ax.set_ylabel('PC'+str(pc_y+1)+' ('+str(round(pca.explained_variance_ratio_[pc_y]*100,2))+'%)')
        ax.set_zlabel('PC'+str(pc_z+1)+' ('+str(round(pca.explained_variance_ratio_[pc_z]*100,2))+'%)')

    return ax, pca

def plot_umap(ax, data, cases, cmap=cm.get_cmap('viridis'), s=20, alpha=.8, umap_params={}):
    """
    This function plots the Uniform Manifold Approximation and Projection (UMAP) of the protein data.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on.
        data (pandas.DataFrame): The protein data to plot.
        cases (list): The cases to include in the plot.
        cmap (matplotlib.colors.Colormap, optional): The colormap to use for the plot. Defaults to 'viridis'.
        s (int, optional): The size of the points in the plot. Defaults to 20.
        alpha (float, optional): The transparency of the points in the plot. Defaults to 0.8.
        umap_params (dict, optional): A dictionary of parameters to pass to the UMAP function. 
            Possible keys are 'n_neighbors', 'min_dist', 'n_components', 'metric', and 'random_state'. 
            Defaults to an empty dictionary, in which case the default UMAP parameters are used.

    Returns:
        ax (matplotlib.axes.Axes): The axes with the plot.
        fit_umap (umap.UMAP): The fitted UMAP object.

    Raises:
        AssertionError: If 'n_components' is 3 but the axes is not a 3D projection.
    """
    default_umap_params = {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 2, 'metric': 'euclidean', 'random_state': 42}
    umap_param = {**default_umap_params, **(umap_params if umap_params else {})}
    
    if umap_param['n_components'] == 3:
        assert ax.name == '3d', "The ax must be a 3D projection, please define projection='3d'"

    dict_data = utils.get_abundance(data, cases, abun_type='raw')

    # make stack of all abundance data
    X = np.hstack([np.array(dict_data[list(dict_data.keys())[i]]) for i in range(len(dict_data))])
    X = X.T

    # make sample array
    y = np.hstack([np.repeat(i, dict_data[list(dict_data.keys())[i]].shape[1]) for i in range(len(dict_data))])
    print(f'BEFORE: Number of samples|Number of proteins: {X.shape}')

    # remove columns that contain nan in X
    X = X[:, ~np.isnan(X).any(axis=0)]
    print(f'AFTER: Number of samples|Number of proteins: {X.shape}')
    # X = np.log2(X+1)

    Xnorm = (X - X.mean(axis=0)) / X.std(axis=0)

    fit_umap = umap.UMAP(**umap_param)
    Xt = fit_umap.fit_transform(Xnorm)
    # Xt = pca.fit_transform(Xnorm)

    if umap_param['n_components'] == 1:
        ax.scatter(Xt[:,0], range(len(Xt)), c=y, cmap=cmap)
        ax.set_xlabel('UMAP 1')
    if umap_param['n_components'] == 2:
        ax.scatter(Xt[:,0], Xt[:,1], c=y, cmap=cmap)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    if umap_param['n_components'] == 3:
        ax.scatter(Xt[:,0], Xt[:,1], Xt[:,2], c=y, cmap=cmap)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')

    return ax, fit_umap

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

def plot_volcano(ax, data, cases, gene_list=5, color=None, alpha=0.5, pval=0.05, log2fc=1, linewidth=0.5, fontsize = 8):
    """
    Plot a volcano plot on the given axes.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to plot.
    data (pandas.DataFrame): The data to plot.
    cases (list): The cases to consider.
    gene_list (int or list): The genes to highlight. If an int, the top and bottom n genes are shown. If a list, only those genes are shown. Can also accept list with 2 numbers to show top and bottom n genes [top, bottom].
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
    >>> ax, volcano_df = plot_volcano(ax, data, cases, log2fc=0.5, pval=0.05, alpha=0.5, fontsize=6, gene_list=name_list);
    
    >>> # Create custom artists
    >>> upregulated = Line2D([0], [0], marker='o', color='w', label='Upregulated', markerfacecolor='red', markersize=6)
    >>> downregulated = Line2D([0], [0], marker='o', color='w', label='Downregulated', markerfacecolor='blue', markersize=6)
    >>> no_significance = Line2D([0], [0], marker='o', color='w', label='No significance', markerfacecolor='grey', markersize=6)

    >>> # Add the legend
    >>> plt.legend(handles=[upregulated, downregulated, no_significance], loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0., frameon=True, prop={'size': 7})
    >>> plt.show()
    """
    volcano_df = utils.get_protein_DE(data, cases)
    volcano_df['p_value'] = volcano_df['p_value'].replace([np.inf, -np.inf], np.nan)
    volcano_df = volcano_df.dropna(subset=['p_value'])
    volcano_df['-log10(p_value)'] = -np.log10(volcano_df['p_value'])
    volcano_df['significance_score'] = volcano_df['-log10(p_value)'] * volcano_df['log2fc']

    volcano_df['significance'] = 'not significant'
    volcano_df.loc[(volcano_df['p_value'] < pval) & (volcano_df['log2fc'] > log2fc), 'significance'] = 'upregulated'
    volcano_df.loc[(volcano_df['p_value'] < pval) & (volcano_df['log2fc'] < -log2fc), 'significance'] = 'downregulated'

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

    if isinstance(gene_list, int):
        upregulated = volcano_df[volcano_df['significance'] == 'upregulated'].sort_values('significance_score', ascending=False)
        downregulated = volcano_df[volcano_df['significance'] == 'downregulated'].sort_values('significance_score', ascending=True)
        
        n_upregulated = min(gene_list, len(upregulated))
        n_downregulated = min(gene_list, len(downregulated))
    
        label_df = pd.concat((upregulated.head(n_upregulated), downregulated.head(n_downregulated)))
    elif isinstance(gene_list, list):
        if len(gene_list) == 2 and all(isinstance(i, int) for i in gene_list):
            upregulated = volcano_df[volcano_df['significance'] == 'upregulated'].sort_values('significance_score', ascending=False)
            downregulated = volcano_df[volcano_df['significance'] == 'downregulated'].sort_values('significance_score', ascending=True)

            n_upregulated = min(gene_list[0], len(upregulated))
            n_downregulated = min(gene_list[1], len(downregulated))
        
            label_df = pd.concat((upregulated.head(n_upregulated), downregulated.head(n_downregulated)))
        else:
            label_df = volcano_df.loc[volcano_df.index.isin(gene_list)]
    else:
        raise ValueError('label_list must be an int or a list')

    texts = []
    for i in range(len(label_df)):
        txt = plt.text(x = label_df.iloc[i]['log2fc'],
                            y = label_df.iloc[i]['-log10(p_value)'],
                            s = label_df.index[i],
                            fontsize = fontsize,
                            weight = 'bold')
        
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
        texts.append(txt)
    adjust_text(texts, arrowprops = dict(arrowstyle = '->', color = 'k', zorder = 5))

    return ax, volcano_df

# VERIFY FUNCTIONS BELOW
def plot_rankquant(ax,data,cases,cmap=['Blues'],color=['blue'],s=20,alpha=0.2,calpha=1):
    # extract columns that contain the abundance data for the specified method and amount
    for j in range(len(cases)):
        vars = ['Abundance: '] + cases[j]
        cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]
        # concat elements 1 till end of vars into one string
        append_string = '_'.join(vars[1:])

        # average abundance of proteins across these columns, ignoring NaN values
        data['Average: '+append_string] = data[cols].mean(axis=1, skipna=True)
        data['Stdev: '+append_string] = data[cols].std(axis=1, skipna=True)

        # sort by average abundance
        data.sort_values(by=['Average: '+append_string], ascending=False, inplace=True)

        # add rank number
        data['Rank: '+append_string] = np.arange(1, len(data)+1)

        nsample = len(cols)

        # merge all abundance columns into one column
        X = np.zeros((nsample*len(data)))
        Y = np.zeros((nsample*len(data)))
        Z = np.zeros((nsample*len(data))) # pdf value based on average abundance
        for i in range(nsample):
            X[i*len(data):(i+1)*len(data)] = data[cols[i]].values
            Y[i*len(data):(i+1)*len(data)] = data['Rank: '+append_string].values
            mu = np.log10(data['Average: '+append_string].values)
            std = np.log10(data['Stdev: '+append_string].values)
            z = (np.log10(X[i*len(data):(i+1)*len(data)]) - mu)/std
            z = np.power(z,2)
            Z[i*len(data):(i+1)*len(data)] = np.exp(-z*70)

        # remove NaN values
        Z = Z[~np.isnan(X)]
        Y = Y[~np.isnan(X)]
        X = X[~np.isnan(X)]
        # plot

        ax.scatter(data['Rank: '+append_string], data['Average: '+append_string], marker='.', color=color[j], alpha=calpha)
        ax.scatter(Y, X, c=Z, marker='.',cmap=cmap[j], s=s,alpha=alpha)
        # set y axis to log
        ax.set_yscale('log')

    return ax

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

def mark_rankquant(plot,data,names,cases,color='red',s=10,alpha=1,show_names=True):
    for j in range(len(cases)):
        vars = ['Abundance: '] + cases[j]

        # concat elements 1 till end of vars into one string
        append_string = '_'.join(vars[1:])
        
        for i, txt in enumerate(names):
            x = data['Average: '+append_string][data['Accession']==txt].values[0]
            y = data['Rank: '+append_string][data['Accession']==txt].values[0]
            if show_names:
                plot.annotate(txt, (y,x), xytext=(y+10,x*1.1), fontsize=8)
            plot.scatter(y,x,marker='o',color=color,s=s, alpha=alpha)
    return plot

def plot_raincloud(ax,data,cases,color=['blue'],boxcolor='black',linewidth=0.5):
    # extract columns that contain the abundance data for the specified method and amount
    data_X = []
    for j in range(len(cases)):
        vars = ['Abundance: '] + cases[j]
        cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]
        # concat elements 1 till end of vars into one string
        append_string = '_'.join(vars[1:])

        # average abundance of proteins across these columns, ignoring NaN values
        data['Average: '+append_string] = data[cols].mean(axis=1, skipna=True)
        data['Stdev: '+append_string] = data[cols].std(axis=1, skipna=True)

        # sort by average abundance
        data.sort_values(by=['Average: '+append_string], ascending=False, inplace=True)

        # add rank number
        data['Rank: '+append_string] = np.arange(1, len(data)+1)

        nsample = len(cols)

        # merge all abundance columns into one column
        X = np.zeros((nsample*len(data)))
        for i in range(nsample):
            X[i*len(data):(i+1)*len(data)] = data[cols[i]].values

        X = X[~np.isnan(X)]
        X = np.log10(X)

        data_X.append(X)
        
    # plot
    # Boxplot data
    bp = ax.boxplot(data_X, positions=np.arange(1,len(cases)+1)-0.06, widths=0.1, patch_artist = True,
                    flierprops=dict(marker='o', alpha=0.2, markersize=2, markerfacecolor=boxcolor, markeredgecolor=boxcolor),
                    whiskerprops=dict(color=boxcolor, linestyle='-', linewidth=linewidth),
                    medianprops=dict(color=boxcolor, linewidth=linewidth),
                    boxprops=dict(facecolor='none', color=boxcolor, linewidth=linewidth),
                    capprops=dict(color=boxcolor, linewidth=linewidth))

    # Create a list of colors for the violin plots based on the number of features you have
    violin_colors = ['thistle', 'orchid']

    # Violinplot data
    vp = ax.violinplot(data_X, points=500, vert=True, positions=np.arange(1,len(cases)+1)+0.06,
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

def mark_raincloud(plot,data,prot_names,cases,lowest_index=0,color='red',s=10,alpha=1,show_names=True):
    for j in range(len(cases)):
        vars = ['Abundance: '] + cases[j]
        append_string = '_'.join(vars[1:])
                
        for i, txt in enumerate(prot_names):
            y = np.log10(data['Average: '+append_string][data['Accession']==txt].values[0])
            x = lowest_index + j + .14 + 0.8
            plot.scatter(x,y,marker='o',color=color,s=s, alpha=alpha)
    return plot
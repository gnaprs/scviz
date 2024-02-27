import numpy as np
import pandas as pd
import re
import utils

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
# from matplotlib.ticker import MaxNLocator
import matplotlib.collections as clt
from upsetplot import plot, generate_counts, from_contents, query, UpSet
import seaborn as sns
sns.set_theme(context='paper', style='ticks')

# --- for marion
# Create a list of colors
colors = ['#FC6908', '#00AEE8', '#9D9D9D', '#6EDC00', '#F4D03F', 'red', '#A454C7']
cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)
palette = sns.color_palette(colors)
# ---

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

def plot_pca(ax, data, cases, cmap=cmap, s=20, alpha=0.2, plot_pc=[0,1]):

    from sklearn.decomposition import PCA
    dict_data = utils.return_abundance(data, cases, abun_type='raw')

    pc_x=plot_pc[0]
    pc_y=plot_pc[1]

    # make stack of all abundance data
    X = np.hstack([np.array(dict_data[list(dict_data.keys())[i]][0]) for i in range(len(dict_data))])
    X = X.T

    # make sample array
    y = np.hstack([np.repeat(i, dict_data[list(dict_data.keys())[i]][0].shape[1]) for i in range(len(dict_data))])

    # number of samples by different proteins
    print(X.shape, y.shape)

    # remove columns that contain nan in X
    X = X[:, ~np.isnan(X).any(axis=0)]
    print(X.shape, y.shape)
    X = np.log2(X+1)

    Xnorm = (X - X.mean(axis=0)) / X.std(axis=0)

    pca = PCA()
    Xt = pca.fit_transform(Xnorm)
  
    ax.scatter(Xt[:,pc_x], Xt[:,pc_y], c=y, cmap=cmap, s=s, alpha=alpha)
    ax.set_xlabel('PC'+str(pc_x+1)+' ('+str(round(pca.explained_variance_ratio_[pc_x]*100,2))+'%)')
    ax.set_ylabel('PC'+str(pc_y+1)+' ('+str(round(pca.explained_variance_ratio_[pc_y]*100,2))+'%)')

    return ax, pca

def plot_pca_scree(ax, pca):
    PC_values = np.arange(pca.n_components_) + 1
    ax.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    ax.set_title('Scree Plot')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    
    return ax

def plot_heatmap_annotated(ax,data,cases,genes, cmap=plt.cm.seismic,norm_values=[4,5.5,7],linewidth=.5,annotate=True, square=False,xticklabels=True):
        # extract columns that contain the abundance data for the specified method and amount
    for j in range(len(cases)):
        vars = ['Abundance: '] + cases[j]
        append_string = '_'.join(vars[1:])
        cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]

        # average abundance of proteins across these columns, ignoring NaN values
        data['Average: '+append_string] = data[cols].mean(axis=1, skipna=True)
        data['Stdev: '+append_string] = data[cols].std(axis=1, skipna=True)

        print(append_string)

    # find the number for the average column  of all cases
    case_col=[]
    for i in range(len(cases)):
        case_col.append(data.columns.get_loc('Average: '+'_'.join(cases[i][:])))

    data = data.copy()
    # only keep rows where genes is in "Gene Symbol"
    data = data[data['Gene Symbol'].isin(genes)]
    # set gene symbol as index
    data.set_index('Gene Symbol', inplace=True)
    print(len(data))
    # only keep columns that are in case_col
    data = data.iloc[:,case_col]
    # drop rows where all values are NaN
    data = data.dropna(how='all')
    
    # log10 data
    data_log10 = np.log10(data)

    mid_norm = mcolors.TwoSlopeNorm(vmin=norm_values[0], vcenter=norm_values[1], vmax=norm_values[2])

    ax = sns.heatmap(data_log10, yticklabels=True, square=square, annot=annotate, linewidth=linewidth, cmap=cmap, norm=mid_norm, cbar_kws={'label': 'Abundance (AU)'})
    # ax = sns.heatmap(data, yticklabels=True, square=square, annot=annotate, linewidth=linewidth, cmap=cmap, norm=LogNorm(10**3.5), cbar_kws={'label': 'Abundance (AU)'})

    return ax

def plot_abundance(ax,data,cases,cmap=['Blues'],color=['blue'],s=20,alpha=0.2,calpha=1):
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

def mark_abundance_protein(plot,data,names,cases,color='red',s=10,alpha=1,show_names=True):
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
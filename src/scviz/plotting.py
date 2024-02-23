import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

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
    sig = 'n.s.'

    # check variable type of pval
    if isinstance(pval, float):
        if pval > 0.05:
            sig = 'n.s.'
        elif pval < 0.05 and pval > 0.01:
            sig = '*'
        elif pval < 0.01 and pval > 0.001:
            sig = '**'
        elif pval < 0.001 and pval > 0.0001:
            sig = '***'
        elif pval < 0.0001:
            sig = '****'
    else:
        sig = pval

    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
    ax.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color=col)

def cv_bnm_plot(ax,data,cases,color=['blue']):
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

def get_cv(data, cases, variables=['region', 'amt'], sharedPeptides = False):
    """
    Calculate the coefficient of variation (CV) for each case in the given data.

    Parameters:
    - data: pandas DataFrame
        The input data containing the CV values.
    - cases: list of lists
        The cases to calculate CV for. Each case is a list of values corresponding to the variables.
    - variables: list, optional
        The variables to consider when calculating CV. Default is ['region', 'amt'].

    Returns:
    - cv_df: pandas DataFrame
        The DataFrame containing the CV values for each case, along with the corresponding variable values.
    """
    
    # check if the len of each element in cases have the same length as len(variables), else throw error message
    if not all(len(cases[i]) == len(variables) for i in range(len(cases))):
        print("Error: length of each element in cases must be equal to length of variables")
        return
    
    # make dataframe for each case with all CV values
    cv_df = pd.DataFrame()

    if sharedPeptides:
        print('test')


    else:
        for j in range(len(cases)):
            vars = ['CV'] + cases[j]
            cols = [col for col in data.columns if all([re.search(r'\b{}\b'.format(var), col) for var in vars])]
            nsample = len(cols)

            # merge all CV columns into one column
            X = np.zeros((nsample*len(data)))
            for i in range(nsample):
                X[i*len(data):(i+1)*len(data)] = data[cols[i]].values
            # remove nans
            X = X[~np.isnan(X)]/100

            # add X to cur_df, and append case info of enzyme, method and amt to each row
            cur_df = pd.DataFrame()
            cur_df['cv'] = X
            for i in range(len(variables)):
                cur_df[variables[i]] = cases[j][i]
            
            # append cur_df to cv_df
            cv_df = cv_df.append(cur_df, ignore_index=True)

    return cv_df
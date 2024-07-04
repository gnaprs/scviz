import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from Bio import SeqIO
import plotly as px
import adjustText
import warnings
import textwrap
import requests
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

#genedown = df_de[df_de['hit'] == 'Downregulated'].index
#geneup = df_de[df_de['hit'] == 'Upregulated'].index
#genehit = df_de[df_de['hit'] != 'Insignificant'].index
#Upregulated.append(set(geneup))
#Downregulated.append(set(genedown))
#print(geneup)
#import_functions.get_string_network(geneup,f'ora\\Upregulated_{sample_name}_{cells}_interactome')
#import_functions.get_string_network(genedown,f'ora\\Downregulated_{sample_name}_{cells}_interactome')
#import_functions.get_string_network(genehit,f'ora\\Hit_{sample_name}_{cells}_interactome')
#anno_up = import_functions.get_string_annotation(geneup,universe)
#print(anno_up)
#anno_up.to_csv(f'ora\\Upregulated_{sample_name}_{cells}_annotation.csv',index = False)

def volcano_plot(data,figname,fc_column,p_column,
                 fc_cutoff = [-1,1],
                 p_cutoff = 0.05,
                 colors = ['navy','darkred','silver'],
                 alpha = 0.8,
                 font = None,
                 figsize = (6,6),
                 gridstyle = None,
                 symmetry = True,
                 labels = [],
                 label_column = None,
                 labelfont = None,
                 highlight_label = False,
                 title = 'Volcano plot',
                 label_sig = True,
                 explode = (0.01,0.02),
                 force = (0.1,0.2),
                 pull = (0.01,0.02),
                 bbox_props = None,
                 arrowprops = None,
                 gridline_x = True,
                 gridline_y = True):
    
    print('checking input ...',end = '\t')
    if font == None:
        font = {"family" : "Arial",
                "weight" : "bold",
                "size" : 16}

    if gridstyle == None:
        gridstyle = {'color' : 'silver',
                    'linestyle' : '--',
                    'linewidth' : 1}

    if labelfont == None:
        labelfont = {"family" : "Arial",
                    "weight" : "bold",
                    "size" : 10}

    if bbox_props == None:
        bbox_props = dict(boxstyle='round', fc='w', ec='k', alpha=1, lw = 0.5)

    if arrowprops == None:
        arrowprops = dict(arrowstyle="-", color='k', lw=0.5)
    print('done')

    print('-log10 transformation of the p-value ...',end = '\t')
    data['log_p'] = data[p_column].apply(lambda x: -np.log10(x))
    print('done')

    print('select up/down regulated proteins ...',end = '\t')
    dsig = data[data[p_column] < p_cutoff]
    dup = dsig[dsig[fc_column] < fc_cutoff[0]]
    ddown = dsig[dsig[fc_column] > fc_cutoff[1]]
    dus = data[data[p_column] > p_cutoff]
    dmid = dsig[dsig[fc_column].between(fc_cutoff[0],fc_cutoff[1])]
    geneup = dup[label_column].to_list()
    genedown = ddown[label_column].to_list()
    genesig = geneup + genedown
    uniup = dup.index
    unidown = ddown.index
    if label_sig:
        labels = list(set(labels) & set(genesig))
    print('done')

    print('setting ylim and xlim for plotting ...',end = '\t')
    ylim = 1.2*data['log_p'].max()
    xlim = [1.2*data[fc_column].min(),1.2*data[fc_column].max()]
    if symmetry:
        xl = np.max([abs(xlim[0]),abs(xlim[1])])
        xlim = [-xl,xl]
    print('done')

    print('define figure and rc ...',end = '\t')
    fig = plt.figure(figsize = figsize,dpi = 600)
    plt.rc('font',**font)
    print('done')

    plt.grid(True,linestyle = '--',color = 'silver',linewidth = 1,alpha = 0.2,zorder = 0)
    print('plotting scatter plot ...',end = '\t')
    plt.scatter(dup[fc_column],dup['log_p'],color = colors[0],s = 12,alpha = alpha,zorder = 1,edgecolors='none')
    plt.scatter(ddown[fc_column],ddown['log_p'],color = colors[1],s = 12,alpha = alpha,zorder = 1,edgecolors='none')
    plt.scatter(dus[fc_column],dus['log_p'],color = colors[2],s = 6,alpha = alpha,zorder = 0,edgecolors='none')
    plt.scatter(dmid[fc_column],dmid['log_p'],color = colors[2],s = 6,alpha = alpha,zorder = 0,edgecolors='none')
    print('done')

    print('add gridline ...',end='\t')
    if gridline_x:
        plt.plot(xlim,[-np.log10(p_cutoff),-np.log10(p_cutoff)],zorder = 0,**gridstyle)
    if gridline_y:
        plt.plot([fc_cutoff[0],fc_cutoff[0]],[0,ylim],zorder = 0,**gridstyle)
        plt.plot([fc_cutoff[1],fc_cutoff[1]],[0,ylim],zorder = 0,**gridstyle)
    print('done')

    print('add labels ...',end = '\t')
    texts = []
    plt.text(0.05,0.87,len(dup),ha = 'center',va = 'top',transform = plt.gca().transAxes,fontdict = font)
    plt.text(0.95,0.87,len(ddown),ha = 'center',va = 'top',transform = plt.gca().transAxes,fontdict = font)
    if label_column != None:
        dsel = data[data[label_column].isin(labels)]
        if highlight_label:
            plt.scatter(dsel[fc_column],dsel['log_p'],color = 'black',s = 12,alpha = alpha,zorder = 2,edgecolors='none')
        for fc,p,l in zip(dsel[fc_column],dsel['log_p'],dsel[label_column]):
            texts.append(plt.text(fc,p,l,fontdict = labelfont,zorder = 1,bbox = {'boxstyle':'round', 'facecolor':'white','edgecolor':'black'}))
    adjustText.adjust_text(texts,x = data[fc_column],y = data['log_p'],
                           avoid_self=True,time_lim = 10,force_static = force,force_text = force,force_explode = explode,force_pull=pull,
                           expand = (1.5,1.5),min_arrow_len=5,arrowprops=arrowprops)
    print('done')

    print('save the figure ...',end = '\t')
    ax = plt.gca()
    ax.spines["top"].set_linewidth(1)
    ax.spines["top"].set_color('black')
    ax.spines["right"].set_linewidth(1)
    ax.spines["right"].set_color('black')
    ax.spines["left"].set_linewidth(1)
    ax.spines["left"].set_color('black')
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["bottom"].set_color('black')
    ax.add_patch(patches.Rectangle((0,0.90),width = 1,height = 0.10,color = 'black',transform = ax.transAxes,zorder = 2))
    plt.text(0.5,0.95,title,fontdict = {'family':'Arial','weight':'bold','size':16},color = 'white',transform = ax.transAxes,va = 'center',ha = 'center',zorder = 3)
    ax.tick_params(axis = "both",which = "major",width = 2)
    plt.xlabel('log2 Fold Change')
    plt.ylabel('-log10 p-value')
    plt.savefig(f'{figname}.png')
    print('done')

    return fig,uniup,unidown

def dotplot(data,plotname):
    data = data[data['FWER p-val'] < 0.05]
    data = data.sort_values(by = 'FWER p-val', ascending = False)
    if len(data) > 15:
        data = data.head(15)
    elif len(data) < 1:
        return 0
    data = data.sort_values(by = 'ES', ascending = False)
    data['wrapTerm'] = data['Term'].apply(lambda x: textwrap.fill(x, width=30))
    data['size'] = data['Tag %'].apply(lambda x: float(x.split('/')[0])/float(x.split('/')[1])*100)
    font = {'family' : 'Arial',
            'weight' : 'heavy',
            'size'   : 12}
    if len(data) < 5:
        height = 3.5
        h = 0.3
    else:
        height = len(data)*0.7
        h = 1.0/len(data)
    f, ax = plt.subplots(figsize=(10, height))
    plt.grid(True,linestyle = '--',color = 'silver',linewidth = 1,alpha = 0.2,zorder = 0)
    # sns.despine(left=True, bottom=True)
    sns.scatterplot(x="ES", y="wrapTerm", hue="FWER p-val", size='size', sizes=(50, 200), data=data, ax=ax,legend = None,palette = 'rocket')
    mapper = plt.cm.ScalarMappable(cmap='rocket', norm=plt.Normalize(vmin=0, vmax=0.05))
    cax = f.add_subplot(212,position=[0.8, 0.4-1/height, 0.03, 1/height])
    plt.colorbar(mapper, cax=cax, orientation='vertical',ticks = [0,0.05])
    font = {'family' : 'Arial',
            'weight' : 'bold',
            'size'   : 14}
    cax.set_title('FDR',fontdict=font,y = 1.15)
    ax1 = f.add_subplot(211,position=[0.8, 0.55, 0.05, 1/height])
    ax1.get_xaxis().set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    sns.scatterplot(x = [0,0,0,0],y = [0,1,2,3],size = [50,100,150,200],sizes=(50, 200),legend = None, ax = ax1,color = 'black',zorder = 1)
    ax1.set_ylim([-0.5,3.5])
    ax1.set_yticks([0,1,2,3],['50','100','150','200'])
    ax1.yaxis.set_label_position("right")
    ax1.set_title('Size',fontdict=font)
    ax.set_xlabel('Enrichment Score')
    ax.set_ylabel('')
    ax.set_position([0.4, h, 0.3, 0.9 - h])
    plt.savefig(plotname)
    plt.close()

def dotplot_ora(data,plotname):
    data = data[data['fdr'] < 0.05]
    data = data.sort_values(by = 'fdr', ascending = False)
    data['GeneRatio'] = data['number_of_genes'] / data['number_of_genes_in_background']
    if len(data) > 15:
        data = data.head(15)
    elif len(data) < 1:
        return 0
    data = data.sort_values(by = 'GeneRatio', ascending = False)
    data['wrapTerm'] = data.apply(lambda x: textwrap.fill(x['description'] + ' ' + x['term'], width=30),axis = 1)
    data['size'] = data['number_of_genes']
    font = {'family' : 'Arial',
            'weight' : 'heavy',
            'size'   : 12}
    # sns.set(style="whitegrid")
    if len(data) < 5:
        height = 3.5
        h = 0.3
    else:
        height = len(data)*0.7
        h = 1.0/len(data)
    f, ax = plt.subplots(figsize=(10, height))
    plt.grid(True,linestyle = '--',color = 'silver',linewidth = 1,alpha = 0.2,zorder = 0)
    # sns.despine(left=True, bottom=True)
    sns.scatterplot(x="GeneRatio", y="wrapTerm", hue="fdr", size='size', sizes=(50, 200), data=data, ax=ax,legend = None,palette = 'rocket',zorder = 1)
    mapper = plt.cm.ScalarMappable(cmap='rocket', norm=plt.Normalize(vmin=0, vmax=0.05))
    cax = f.add_subplot(212,position=[0.8, 0.4-1/height, 0.03, 1/height])
    plt.colorbar(mapper, cax=cax, orientation='vertical',ticks = [0,0.05])
    font = {'family' : 'Arial',
            'weight' : 'bold',
            'size'   : 14}
    cax.set_title('FDR',fontdict=font,y = 1.15)
    ax1 = f.add_subplot(211,position=[0.8, 0.55, 0.05, 1/height])
    ax1.get_xaxis().set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    sns.scatterplot(x = [0,0,0,0],y = [0,1,2,3],size = [50,100,150,200],sizes=(50, 200),legend = None, ax = ax1,color = 'black')
    ax1.set_ylim([-0.5,3.5])
    ax1.set_yticks([0,1,2,3],['50','100','150','200'])
    ax1.yaxis.set_label_position("right")
    ax1.set_title('Size',fontdict=font)
    ax.set_xlabel('Rich Factor')
    ax.set_ylabel('')
    ax.set_position([0.4, h, 0.3, 0.9 - h])
    plt.savefig(plotname)
    plt.close()

def get_string_id(gene,species = 9606):
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "tsv-no-header"
    method = "get_string_ids"
    
    params = {
        "identifiers" : "\r".join(gene),
        "species" : species, 
        "limit" : 1, 
        "echo_query" : 1,
    }
    
    request_url = "/".join([string_api_url, output_format, method])
    results = requests.post(request_url, data=params)
    s_id = []
    for line in results.text.strip().split("\n"):
        l = line.split("\t")
        try:
            string_id = l[2]
            s_id.append(string_id)
        except:
            continue
    
    return s_id

def get_string_annotation(gene,universe,species = 9606):
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "json"
    method = "enrichment"

    params = {
        "identifiers" : "%0d".join(gene),
        "background_string_identifiers": "%0d".join(universe),
        "species" : species
    }
    
    request_url = "/".join([string_api_url, output_format, method])
    results = requests.post(request_url, data=params)
    print(results.text)
    try:
        annotation = pd.read_json(results.text)
    except:
        annotation = pd.DataFrame()
    return annotation

def get_string_ppi(gene,universe,species = 9606):
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "json"
    method = "ppi_enrichment"

    params = {
        "identifiers" : "%0d".join(gene),
        "background_string_identifiers": "%0d".join(universe),
        "species" : species
    }
    
    request_url = "/".join([string_api_url, output_format, method])
    results = requests.post(request_url, data=params)
    print(results.text)
    try:
        annotation = pd.read_json(results.text)
    except:
        annotation = pd.DataFrame()
    return annotation

def get_string_network(gene,comparison,species = 9606):
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "highres_image"
    method = "network"
    
    if len(gene) <= 10:
        hide_label = 0
    else:
        hide_label = 1
    
    params = {
        "identifiers" : "%0d".join(get_string_id(gene)),
        "species" : species,
        "required_score" : 700,
        "hide_disconnected_nodes" : hide_label,
        "block_structure_pics_in_bubbles" : 1,
        "flat_node_design" : 1,
        "center_node_labels" : 1
    }
    
    request_url = "/".join([string_api_url, output_format, method])
    response = requests.post(request_url, data=params)
    
    with open(f'{comparison}.png','wb') as file:
        file.write(response.content)
    
    return True


def boxplot_normalization(dff,filename):
    plt.figure(figsize=(8, 4))
    df = dff.copy()
    df['Sample'] = dff['Sample'].map(lambda x: x.split(',')[-1].strip().split('_')[1])
    df = df.sort_values(by = 'Sample',ascending = False)
    keys = df['Sample'].unique()
    color_palette = sns.color_palette('tab10',len(keys))
    palette = dict(zip(keys,color_palette))
    ax = sns.boxplot(data = df, x = 'File', y = 'Abundance', hue = 'Sample', palette = palette, width = 0.6, legend = False, log_scale = True)
    ticklabels = [item.get_text().split(',')[-1].strip().split('_')[1] for item in ax.get_xticklabels()]
    ax.set_xticklabels(ticklabels, rotation=60)
    ax.set_position([0.1,0.25,0.8,0.7])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'{filename}.png')
    plt.close()

def pca_plot(df,filename):
    font = {'family' : 'Arial',
            'weight' : 'bold',
            'size'   : 14}
    plt.rc('font',**font)
    n_components = 10
    pca = PCA(n_components = n_components)
    df = df.fillna(0)
    scaler = StandardScaler()
    df_pca = df.pivot(index = 'Accession', columns = 'File', values = 'Abundance').transpose()
    df_pca = pd.DataFrame(scaler.fit_transform(df_pca),index = df_pca.index,columns = df_pca.columns)
    pca_columns = [f'PC{i}' for i in range(1,n_components + 1)]
    pca_fit = pca.fit(df_pca)
    df_pca = pd.DataFrame(pca_fit.transform(df_pca),index = df_pca.index,columns = pca_columns)
    df_pca['Sample'] = df_pca.index.map(lambda x: x.split(',')[-1].strip().split('_')[1])
    df_pca = df_pca.sort_values(by = 'Sample',ascending = False)
    fig, ax = plt.subplots(figsize = (6,6),dpi = 600)
    keys = df_pca['Sample'].unique()
    color_palette = sns.color_palette('tab10',len(keys))
    palette = dict(zip(keys,color_palette))
    sns.scatterplot(data = df_pca,x = 'PC1',y = 'PC2',hue = 'Sample',s = 500,alpha = 0.4,linewidth = 0,palette = palette,ax = ax)
    plt.xlabel(f'PC1 {pca_fit.explained_variance_ratio_[0]*100:.1f}%')
    plt.ylabel(f'PC2 {pca_fit.explained_variance_ratio_[1]*100:.1f}%')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'{filename}_pca.png')
    plt.close()
    fig, ax = plt.subplots(figsize = (8,4),dpi = 600)
    plt.bar(pca_columns,pca_fit.explained_variance_ratio_,width = 0.6,color = 'white',edgecolor = 'black',linewidth = 2)
    plt.plot(pca_columns,pca_fit.explained_variance_ratio_,color = 'black')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    ax.set_position([0.1,0.25,0.8,0.7])
    plt.savefig(f'{filename}_pca_variance.png')
    plt.close()

def de_analysis(df,cond1,cond2):
    df = df[df['Sample'].isin([cond1,cond2])]
    df_de = df.pivot(index = 'Accession', columns = 'File', values = 'Abundance')
    for group in df.groupby('Accession'):
        tstat,pvalue = ttest_ind(np.log2(group[1][group[1]['Sample'] == cond1]['Abundance']),np.log2(group[1][group[1]['Sample'] == cond2]['Abundance']),nan_policy='omit')
        df_de.loc[group[0],'pvalue'] = pvalue
        df_de.loc[group[0],'t_stat'] = tstat
    df_de['log2fc'] = np.log2(df[df['Sample'] == cond1].groupby('Accession')['Abundance'].mean() / df[df['Sample'] == cond2].groupby('Accession')['Abundance'].mean())
    # rej, df_de['fdr'] = fdrcorrection(df_de['pvalue'])
    return df_de

def plot_abundance(df,poi_list,filename):
    n = int(len(poi_list))
    num = int(np.floor(np.sqrt(n)) + 1)

    if num*(num - 1) >= n:
        ncols = num
        nrows = num - 1
    else:
        ncols = num
        nrows = num
        
    fig, axs = plt.subplots(
        ncols = ncols,nrows = nrows,constrained_layout = True)

    for i in range(0, n):
        y = int(i//num)
        x = int(i%num)
        ax = axs[y,x]
        ax.set_title(poi_list[i],fontdict = {'fontsize': 8, 'fontweight': 'bold'})
        data = df[df['Gene Symbol'] == poi_list[i]].dropna(subset = ['Abundance'])
        # ax.set_yscale('log')
        sns.boxplot(data = data, x = 'Sample', y = 'Abundance',hue = 'Sample', width = 0.6, ax = ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticks([0,df[df['Gene Symbol'] == poi_list[i]]['Abundance'].max()])
        ax.set_yticklabels([0,np.format_float_scientific(df[df['Gene Symbol'] == poi_list[i]]['Abundance'].max(),precision = 1)],
                           fontdict = {'fontsize': 6, 'fontweight': 'normal'})

    nn = 0
    for j in range(0,nrows):
        for i in range(0,ncols):
            ax = axs[j,i]
            if i == 0:
                ax.set_ylabel('Abundance',fontdict = {'fontsize': 8, 'fontweight': 'bold'})
            if j == nrows - 1:
                ax.set_xticklabels(df['Sample'].unique(), rotation=60,fontdict = {'fontsize': 6, 'fontweight': 'bold'})
            nn += 1
            if nn > n:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

    plt.savefig(f'{filename}_abundance.png', dpi = 600)
    plt.close()

from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
import numpy as np
import scanpy as sc
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shap.plots import colors
from shap.plots import summary

def venn(adata, label_key, key_list, cell_types=None, save=False):
    """
    plot venn diagram to show intersection between different methods
    :param adata: ~anndata.AnnData`
                    Annotated Data Matrix.
    :param label_key: basestring
    :param key_list: list
                    some keys from adata.uns['shared_genes'] to plot with each other
    :param cell_types: list
                    name of the expected cell types
    :param save: basestring
                path to save the plot.
    :return:
    """

    cut_name = 17
    plt.figure(figsize=(45, 20))
    if cell_types == None:
        cell_types = np.unique(adata.obs[label_key].values)
    for (i, key) in zip([s for s in range(len(key_list))], key_list):
        names = key.split('+')
        if len(names) == 2:
            shared_genes = adata.uns['shared_genes'][key]['numbers']
            for (j, c_t) in zip([s for s in range(len(cell_types))], cell_types):
                plt.subplot(len(key_list), len(cell_types), j + 1 + i * (len(cell_types)))
                plt.title(c_t, fontdict={'fontsize': 28})
                v = venn2(subsets=shared_genes[c_t], set_labels=(names[0][:cut_name], names[1][:cut_name]))
                if shared_genes[c_t][-1] is not 0:
                    for text in v.set_labels:
                        text.set_fontsize(16)
                    for text in v.subset_labels:
                        text.set_fontsize(20)
                venn2_circles(shared_genes[c_t], linestyle='dotted', alpha=0.1)
            plt.subplots_adjust(wspace=0.8, hspace=0.3)

        elif len(key.split('+')) == 3:
            shared_genes = adata.uns['shared_genes'][key]['numbers']
            for (j, c_t) in zip([s for s in range(len(cell_types))], cell_types):
                plt.subplot(len(key_list), len(cell_types), j + 1 + i * (len(cell_types)))
                plt.title(c_t, fontdict={'fontsize': 28})
                if shared_genes[c_t][0]== shared_genes[c_t][2] and shared_genes[c_t][4]==shared_genes[c_t][6] and shared_genes[c_t][0]==shared_genes[c_t][6] and shared_genes[c_t][0] ==0:
                    genes = [shared_genes[c_t][1], shared_genes[c_t][3], shared_genes[c_t][5]]
                    v = venn2(subsets=genes, set_labels=(names[1][:cut_name], names[2][:cut_name]))
                    venn2_circles(genes, linestyle='dotted', alpha=0.1)
                else:
                    v = venn3(shared_genes[c_t], set_labels=(names[0][:cut_name], names[1][:cut_name], names[2][:cut_name]))
                    venn3_circles(shared_genes[c_t], linestyle='dotted', alpha=0.1, color='grey')
            plt.subplots_adjust(wspace=0.8, hspace=0.3)
    if save:
        plt.savefig(save + ' venn.png', dpi=100)

def heatmap(adata, selected_genes, minv=-np.inf, maxv=np.inf, save=None, label_key='cell_label',
            show_gene_labels=False):
    '''

    :param adata: annotation data
    :param selected_genes: the columns of heatmap
    :param minv: clipping smaller values of adata.X to minv
    :param maxv: clipping bigger values of adata.X to maxv
    :param save: title of plot
    :param label_key: header of label column in adata.obs
    :param show_gene_labels: show or hide feature names from adata.var_names
    :return:
    '''
    new_adata = adata.copy()

    for s in range(new_adata.shape[1]):
        new_adata.X[:, s] = np.clip(new_adata.X[:, s], minv, maxv)

    sc.pl.heatmap(new_adata, selected_genes, groupby=label_key, figsize=(19, 15), save=save,
                  show_gene_labels=show_gene_labels, use_raw=False)


labels = {
    'MAIN_EFFECT': "Relevance main effect value for\n%s",
    'INTERACTION_VALUE': "Relevance interaction value",
    'INTERACTION_EFFECT': "Relevance interaction value for\n%s and %s",
    'VALUE': "Relevance value (impact on model output)",
    'GLOBAL_VALUE': "mean(|Relevance value|) (average impact on model output magnitude)",
    'VALUE_FOR': "Relevance value for\n%s",
    'PLOT_FOR': "Relevance plot for %s",
    'FEATURE': "Feature %s",
    'FEATURE_VALUE': "Log Count Data",
    'FEATURE_VALUE_LOW': "Low",
    'FEATURE_VALUE_HIGH': "High",
    'MODEL_OUTPUT': "Model output value"
}


# inspired by https://github.com/slundberg/shap/


def scplot(adata, method, gene_keys=None, max_display=None, plot_type=None, title=None):
    """Create a relevance summary plot, colored by feature values when they are provided.

    Parameters
    ----------
    adata : `~anndata.AnnData`
                Annotated Data Matrix.
    method : basestring
            interpretability method name
    gene_keys : list
                list of genes
    feature_names : list
                    Names of the features (length # features)

    max_display : int
        How many top features to include in the plot (default is 20)

    plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin",
        or "compact_dot".
        What type of summary plot to produce. Note that "compact_dot" is only used for
        relevance interaction values.
    """
    if gene_keys is None:
        gene_keys = adata.var_names

    features = pd.DataFrame(adata[:, gene_keys].X, columns=gene_keys)
    relevance_values = adata[:, gene_keys].layers[method]
    summary.summary_plot(shap_values=relevance_values, features=features, feature_names=None, max_display=max_display, plot_type=plot_type, title=title)

def shorten_text(text, length_limit):
    if len(text) > length_limit:
        return text[:length_limit - 3] + "..."
    else:
        return text


def scatterPlot(model, adata, node_index, observation):
    """
        Scatter plot

        Predicts the data using model, then plots the values of specified output index against observation.

        Parameters
        ----------
        model
            The model should have a function `predict` (e.g., Keras model).
        adata
            The annotated data matrix.
        node_index
            The output index of which we want the values.`
        observation
            `adata.obs` key which will be used as y coordinate.

    """
    cellLabels = np.array(adata.obs[observation])
    encoded = model.predict(adata.X)[:, node_index]
    sns.scatterplot(encoded, cellLabels)
    plt.xlabel('Hidden Value')
    plt.ylabel(observation)
    plt.show()


def bar(adata, type, methods, label_key, deg=True, n=10, paper=False, save=None, scale=False):
    """

    :param adata: `~anndata.AnnData`
                Annotated Data Matrix.
    :param type: - or +
                if -:
                    (paper & method) not in deg
                if +:
                    if deg = False and paper=False:
                        intersection between methods
                    if deg=True and paper=False:
                        intersection between methods and deg
                    if deg=True and paper=True:
                        intersection between paper and deg and methods
    :param methods: list
                list of mathod names
    :param label_key: basestring
                expected label column in adata.obs
    :param deg: bool
                Differentially Extracted Genes
    :param n: int
                number of captured top genes from each method and deg
    :param paper: bool
                load or not to load paper genes
    :param save:basestring
                path to save the plot
    :param scale:bool
                if scale == True:
                    then we will have num_shared_genes/num_paper_genes for each specific cell type in the bar plot
                else:
                    we would just have  num_shared_genes for each specific cell type in the bar plot
    :return:
    """

    #todo should change the sintax of + and - for more readability
    cell_types = adata.obs[label_key].astype('category').cat.categories

    def _pap_Meth():
        res = {}
        for item in adata.uns['shared_genes'].keys():
            meth_name = None
            for m in methods:
                if meth_name is None and m in item:
                    meth_name = m
                    break
            if ('pap' in item) and (meth_name is not None) and (len(item.split('+')) == 2) and (
                    str(n) == item[-len(str(n)):]):
                res[item] = adata.uns['shared_genes'][item]

        if res is not None:
            return res
        else:
            print("_pap_meth is not working...! maybe you should change the input of bar method")

    def _meth():
        res = {}
        for item in adata.uns['shared_genes'].keys():
            meth_name1 = None
            meth_name2 = None
            for m in methods:
                if (meth_name1 is None) and (m in item):
                    meth_name1 = m

                elif (meth_name2 is None) and (m in item):
                    meth_name2 = m

            if (meth_name1 is not None) and (meth_name2 is not None) and (len(item.split('+')) == 2) and (
                    str(n) == item[-len(str(n)):]):
                res[item] = adata.uns['shared_genes'][item]

        if res is not None:
            return res
        else:
            print("_meth is not working...! maybe you should change the input of bar method")

    def _deg_meth():
        res = {}
        for item in adata.uns['shared_genes'].keys():
            meth_name = None
            for m in methods:
                if meth_name is None and m in item:
                    meth_name = m
                    break
            if ('deg' in item) and (meth_name is not None) and (len(item.split('+')) == 2) and (
                    str(n) == item[-len(str(n)):]):
                res[item] = adata.uns['shared_genes'][item]

        if res is not None:
            return res
        else:
            print("_deg_meth is not working...! maybe you should change the input of bar method")

    def _pap_deg_meth():
        res = {}
        for item in adata.uns['shared_genes'].keys():
            meth_name = None
            for m in methods:
                if meth_name is None and m in item:
                    meth_name = m
                    break
            if ('pap' in item) and ('deg' in item) and (meth_name is not None) and (len(item.split('+')) == 3) and (
                    str(n) == item[-len(str(n)):]):
                res[item] = adata.uns['shared_genes'][item]

        if res is not None:
            return res
        else:
            print("_pap_deg_meth is not working...! maybe you should change the input of bar method")

    def _pap_meth_meth_robust():
        res = {}
        for item in adata.uns['shared_genes'].keys():
            meth_name = None
            for m in methods:
                if meth_name is None and m in item:
                    meth_name = m
                    break
            if ('pap' in item) and ('robust' in item) and (meth_name is not None) and (len(item.split('+')) == 3) and (
                    str(n) == item[-len(str(n)):]):
                res[item] = adata.uns['shared_genes'][item]

        if res is not None:
            return res
        else:
            print("_pap_meth_meth_robust is not working...! maybe you should change the input of bar method")

    def _deg_meth_meth_robust():
        res = {}
        for item in adata.uns['shared_genes'].keys():
            meth_name = None
            for m in methods:
                if meth_name is None and m in item:
                    meth_name = m
                    break
            if ('deg' in item) and ('robust' in item) and (meth_name is not None) and (len(item.split('+')) == 3) and (
                    str(n) == item[-len(str(n)):]):
                res[item] = adata.uns['shared_genes'][item]

        if res is not None:
            return res
        else:
            print("_deg_meth_meth_robust is not working...! maybe you should change the input of bar method")


    def _plot(res, which):
        which = 2 * which - 2
        index = list(res.keys())
        columns = cell_types
        table = np.zeros((len(index), len(columns)))
        df = pd.DataFrame(table, columns=columns, index=index)

        for i, k in enumerate(index) :
            # if scale and which== 4:
            #     df.iloc[i] = (pd.DataFrame(res[k]['numbers']).iloc[which]/ (pd.DataFrame(res[k]['numbers']).iloc[0]+1))
            # else:
            df.iloc[i] = pd.DataFrame(res[k]['numbers']).iloc[which]
        rep_cell_types = np.array(np.repeat(cell_types, table.shape[0])).reshape(
            (table.shape[1], table.shape[0])).T.reshape(-1)
        rep_intersections = np.array(np.repeat(index, table.shape[1])).reshape(-1)
        intersections = table.reshape(-1)
        df = pd.DataFrame(np.array([rep_cell_types, rep_intersections, intersections]).T,
                          columns=['cell_types', 'methods', 'values'])

        # todo tihs if should not work for all different situations!!
        if scale:
            for i in range(df.shape[0]):
                if df['values'][i] > 0:
                    df['values'][i] /= len(adata.uns['top_genes']['paper_genes'][df['cell_types'][i]])

        subcat = 'methods'
        cat = 'cell_types'
        val = 'values'
        grouped_barplot(df, cat, subcat, val, filename='barplot', legend=True)

        # fig, ax = plt.subplots(figsize=(30, 15))
        # ax.set_xticklabels(rep_intersections)
        # sns.barplot(ax=ax, x="cell_types", y="values", hue="methods", data=df, palette='bright')
        return res, df

    if type == '+':
        if (not deg) and paper:
            _plot(_pap_Meth(), 2)
        elif (not deg) and (not paper):
            _plot(_meth(), 2)
        elif deg and (not paper):
            _plot(_deg_meth(), 2)

    elif type == '-':
        _plot(_pap_deg_meth(), 3)



def grouped_barplot(df, cat, subcat, val, filename, legend=False, offset=0.375):
    """
    generates a grouped barplot
    :param df: dataframe of the data
    :param cat: categories like cell_types
    :param subcat: subcategories like interpretability methods
    :param val: values
    :param filename: path for saving plot
    :param legend: legend of the plot
    :param offset: for adjusting columns
    :return:
    """
    plt.close("all")
    import matplotlib
    import os
    matplotlib.rc('ytick', labelsize=25)
    matplotlib.rc('xtick', labelsize=30)
    u = df[cat].unique()
    x_pos = np.arange(0, 5 * len(u), 5)
    subx = df[subcat].unique()
    plt.figure(figsize=(18, 14))
    for i, gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        b = plt.bar(x_pos + i / 1.25, dfg[val].values, capsize=10, alpha=0.95, label=f"{gr}")
        a = dfg[val].values
        plt.plot(x_pos + i / 1.25, a.T, '.', color='black', alpha=0.5)

    plt.ylabel("Number of shared genes ", fontsize=25)
    plt.xticks(x_pos + offset, u, rotation=90)
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0, prop={'size': 18})
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

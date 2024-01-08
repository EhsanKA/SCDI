import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns

# removed from Nature( maybe the link has changed in the website) HABER_SUPP_LINK = 'https://media.nature.com/original/nature-assets/nature/journal/v551/n7680/extref/nature24489-s5.xlsx'
HABER_SUPP_LINK = 'https://static-content.springer.com/esm/art%3A10.1038%2Fnature24489/MediaObjects/41586_2017_BFnature24489_MOESM5_ESM.xlsx'

def haber_supp(link):
    """

    :param link: link of haber supplementary  table for adjusting cell types
    :return:
    """
    df = pd.read_excel(link, skiprows=5)
    haber = {}
    for c_t in df.columns.tolist():
        genes = df[c_t].values.tolist()
        genes = [item for item in genes if str(item) != 'nan']
        haber[c_t] = genes

    haber['Enterocyte'] = haber['Enterocyte (Proximal)']
    haber['Enterocyte'] += haber['Enterocyte (Distal)']
    del haber['Enterocyte (Proximal)']
    del haber['Enterocyte (Distal)']
    haber['TA'] = {}
    haber['Stem'] = {}
    haber['EP'] = {}

    return haber


def top_deGenes(adata, n_top, label_key='cell_label'):
    """
    extracting n_top genes using wilcoxon method
    :param adata: annotation data
    :param n_top: number of expected genes
    :param label_key: basestring
                    header of observation in adata related to label of cells
    :return: saving top genes for each cell label in adata.uns['top_genes']['deg_'+str(n_top)]
    """
    degens = {}
    sc.tl.rank_genes_groups(adata, groupby=label_key, method='wilcoxon', n_genes=n_top)
    for c_type in adata.uns['label'].keys():
        wilcoxon_genes = adata.uns['rank_genes_groups']['names'][c_type][:n_top]
        degens[c_type] = wilcoxon_genes
    adata.uns['top_genes']['deg_'+str(n_top)] = degens
    
    
def top_method_genes(adata, layer_key, n_top, label_key='cell_label'):
    """

    :param adata: annotation data
    :param layer_key: name of interpretability method
    :param n_top: number of expected top genes
    :param label_key: basestring
                    header of observation in adata related to label of cells
    :return: saving top genes for each cell label in adata.uns['top_genes'][layer_key+ '_'+str(n_top)]
    """
    relevance_data = adata.layers[layer_key]
    all_genes_per_cell_type = {}
    for c_t in adata.uns['label'].keys():
        all_genes_per_cell_type[c_t] = []

    # extracting all top n_top genes per sample and integrated theme  for each cell type
    for i in range(relevance_data.shape[0]):
        top_genes_index_per_sample = relevance_data[i,:].argsort()[-n_top:][::-1]
        all_genes_per_cell_type[adata.obs[label_key][i]] += top_genes_index_per_sample.tolist()

    # extracting n_top genes indices per cell type
    top_genes_index_per_cell_type = {}
    for c_t in adata.uns['label'].keys():
        top_genes_index_per_cell_type[c_t] = np.bincount(np.array(all_genes_per_cell_type[c_t])).argsort()[-n_top:][::-1]

    # extracting top genes names per each cell type
    top_genes_per_cell_type = {}
    for c_t in adata.uns['label'].keys():
        top_genes_per_cell_type[c_t] = adata.var_names[top_genes_index_per_cell_type[c_t]]

    adata.uns['top_genes'][layer_key+ '_'+str(n_top)] = top_genes_per_cell_type

    
def calc_intersections(adata, key_list):
    """
    calculating intersection between the lists which their names are in key_list
    :param adata: `~anndata.AnnData`
                Annotated Data Matrix.
    :param key_list: list of two or three strings
                the strings should be in 'adata.uns['top__genes'].keys()
    :return:
                the output would be saved with this format : adata.uns['shared_genes'][first_name+'+'+second_name]['genes'] / ['numbers']
    """

    if len(key_list) ==2:
        k1, k2 = key_list
        l1, l2 = adata.uns['top_genes'][k1], adata.uns['top_genes'][k2]
        genes_p_c= {}
        numbers_p_c = {}
        for k in l1.keys():
            genes = []
            genes.append(set(l1[k]) - set(l2[k]))
            genes.append(set(l2[k]) - set(l1[k]))
            genes.append(set(l1[k]) & set(l2[k]))
            numbers = [len(i) for i in genes]
            genes_p_c[k] = genes
            numbers_p_c[k] = numbers
        
        adata.uns['shared_genes'][k1+'+'+k2] = {}
        adata.uns['shared_genes'][k1+'+'+k2]['genes'] = genes_p_c
        adata.uns['shared_genes'][k1+'+'+k2]['numbers'] = numbers_p_c

    elif len(key_list)==3:
        k1, k2, k3 = key_list
        l1, l2, l3 = adata.uns['top_genes'][k1], adata.uns['top_genes'][k2], adata.uns['top_genes'][k3]
        genes_p_c= {}
        numbers_p_c = {}
        for k in l2.keys():
            genes = []
            genes.append(set(l1[k]) - set(l2[k]) - set(l3[k]))
            genes.append(set(l2[k]) - set(l1[k]) - set(l3[k]))
            genes.append(set(l1[k]) & set(l2[k]) - set(l3[k]))
            genes.append(set(l3[k]) - set(l2[k]) - set(l1[k]))
            genes.append(set(l3[k]) & set(l1[k]) - set(l2[k]))
            genes.append(set(l3[k]) & set(l2[k]) - set(l1[k]))
            genes.append(set(l1[k]) & set(l2[k]) & set(l3[k]))
            numbers = [len(i) for i in genes]
            genes_p_c[k] = genes
            numbers_p_c[k] = numbers
        adata.uns['shared_genes'][k1+'+'+k2+'+'+ k3] = {}
        adata.uns['shared_genes'][k1+'+'+k2+'+'+ k3]['genes'] = genes_p_c
        adata.uns['shared_genes'][k1+'+'+k2+'+'+ k3]['numbers'] = numbers_p_c


# final interperetability function . . .
def finInt(adata, methods, top_gene_list, label_key, paper_genes=None):
    """
    :param adata: annotation data
    :param methods: list of interpretability methods
    :param top_gene_list: list of top expected genes
    :param label_key: basestring
                    header of observation in adata related to label of cells
    :param paper_genes: dict
                    a dictionary which its keys are cell labels and each of them equals to a list of genes
    :return:
                1. creating a dictionary of cell labels
                2. for each method extracts top genes regarding to top_method_genes
                3. for DEG extracting top genes regarding to top_deGenes
                4. calculate intersection between methods
                5. calculate intersection between DEG and mthods
                6. if paper_genes != None:
                    1. calculate intersection between paper and methods
                    2. calculate intersection between paper and DEG
                    3. calculate intersection between paper and DEG and each method
    """

    label_dict = create_dictionary(np.unique(adata.obs[label_key]),
                                   [i for i in range(np.unique(adata.obs[label_key]).shape[0])])
    adata.uns['label'] = label_dict

    if 'shared_genes' not in adata.uns.keys():
        adata.uns['shared_genes'] = {}
    if 'top_genes' not in adata.uns.keys():
        adata.uns['top_genes'] = {}
    for method in methods:
        adata.layers[method] = np.abs(adata.layers[method])
        for i in top_gene_list:
            top_method_genes(adata, method, i, label_key=label_key)
    for i in top_gene_list:
        top_deGenes(adata, i, label_key=label_key)

    # intersection between methods
    if len(methods) >1:
        for i in top_gene_list:
            for ind_meth1 in range(len(methods)):
                for ind_meth2 in range(ind_meth1):
                    calc_intersections(adata, [str(methods[ind_meth1])+'_'+str(i), str(methods[ind_meth2])+'_'+str(i)])

    # intersection between deg and methods
    for i in top_gene_list:
        for ind_meth in range(len(methods)):
            calc_intersections(adata, ['deg_'+str(i), str(methods[ind_meth])+'_'+str(i)])

    if paper_genes is not None:
        adata.uns['top_genes']['paper_genes'] = paper_genes

        # intersection between paper and each method
        for i in top_gene_list:
            for ind_meth in range(len(methods)):
                calc_intersections(adata, ['paper_genes', str(methods[ind_meth]) + '_' + str(i)])

        # intersection between paper and deg
        for i in top_gene_list:
            calc_intersections(adata, ['paper_genes', 'deg_' + str(i)])

        # triple intersection between paper and deg and each method
        for i in top_gene_list:
            for ind_meth in range(len(methods)):
                calc_intersections(adata, ['paper_genes', 'deg_' + str(i), str(methods[ind_meth]) + '_' + str(i)])

def pca(model, adata):
    """
        Predicts the data using model, then computes PCA coordinates.

        Parameters
        ----------
        model
            The model should have a function `predict` (e.g., Keras model).
        adata
            The annotated data matrix.


        Returns
        -------
        adata: anndata.AnnData
            If ``inplace = False`` it returns, or else add fields to `adata`:

            ``.X``
                Same as `adata.X`
            ``.obs``
                Same as `adata.obs`
            ``.obsm['X_pca']``
                 PCA representation of data.
            ``.varm['PCs']``
                 The principal components containing the loadings.
            ``.uns['pca']['variance_ratio']``)
                 Ratio of explained variance.
            ``.uns['pca']['variance']``
                 Explained variance, equivalent to the eigenvalues of the covariance matrix.

    """

    latent = sc.AnnData(model.predict(adata.X), adata.obs)
    sc.tl.pca(latent)
    return latent


def create_dictionary(labels, target_labels):

    """
    making a dictionary from a list to another list
    :param labels: list
                    origin
    :param target_labels: list
                    final
    :return: dictionary
                from first list to second list
    """
    if not isinstance(target_labels, list):
        target_labels = [target_labels]

    dictionary = {}
    labels = [e for e in labels if e not in target_labels]
    for idx, label in enumerate(labels):
        dictionary[label] = idx
    return dictionary



    ## this function is for saving the adata and solving that
    # when we generate the top_genes, it is a list and saving adata gets some problem
    # so we turn the list to a dict and then in the load_adata function we turn it back.
def save_adata(adata):
    bdata = adata.copy()
    for method_i in bdata.uns['top_genes'].keys():
        for c_t in bdata.uns['top_genes'][method_i].keys():
            a = list(bdata.uns['top_genes'][method_i][c_t])
            bdata.uns['top_genes'][method_i][c_t] = {}
            for i in range(len(a)):
                bdata.uns['top_genes'][method_i][c_t][str(i)] = a[i]
    return bdata


def load_adata(adata):
    bdata = adata.copy()

    for method_i in adata.uns['top_genes'].keys():
        for c_t in adata.uns['top_genes'][method_i].keys():
            a = []
            for i in adata.uns['top_genes'][method_i][c_t].keys():
                a.append(bdata.uns['top_genes'][method_i][c_t][i])
            bdata.uns['top_genes'][method_i][c_t] = a
    return bdata


def _hvg_batch(adata, batch_key=None, target_genes=2000, flavor='cell_ranger', n_bins=20, adataOut=False):
    """
    Method to select HVGs based on mean dispersions of genes that are highly
    variable genes in all batches. Using a the top target_genes per batch by
    average normalize dispersion. If target genes still hasn't been reached,
    then HVGs in all but one batches are used to fill up. This is continued
    until HVGs in a single batch are considered.
    """

    adata_hvg = adata if adataOut else adata.copy()

    n_batches = len(adata_hvg.obs[batch_key].cat.categories)

    # Calculate double target genes per dataset
    sc.pp.highly_variable_genes(adata_hvg,
                                flavor=flavor,
                                n_top_genes=target_genes,
                                n_bins=n_bins,
                                batch_key=batch_key)

    nbatch1_dispersions = adata_hvg.var['dispersions_norm'][adata_hvg.var.highly_variable_nbatches >
                                                            len(adata_hvg.obs[batch_key].cat.categories) - 1]

    nbatch1_dispersions.sort_values(ascending=False, inplace=True)

    if len(nbatch1_dispersions) > target_genes:
        hvg = nbatch1_dispersions.index[:target_genes]

    else:
        enough = False
        print(f'Using {len(nbatch1_dispersions)} HVGs from full intersect set')
        hvg = nbatch1_dispersions.index[:]
        not_n_batches = 1

        while not enough:
            target_genes_diff = target_genes - len(hvg)

            tmp_dispersions = adata_hvg.var['dispersions_norm'][adata_hvg.var.highly_variable_nbatches ==
                                                                (n_batches - not_n_batches)]

            if len(tmp_dispersions) < target_genes_diff:
                print(f'Using {len(tmp_dispersions)} HVGs from n_batch-{not_n_batches} set')
                hvg = hvg.append(tmp_dispersions.index)
                not_n_batches += 1

            else:
                print(f'Using {target_genes_diff} HVGs from n_batch-{not_n_batches} set')
                tmp_dispersions.sort_values(ascending=False, inplace=True)
                hvg = hvg.append(tmp_dispersions.index[:target_genes_diff])
                enough = True

    print(f'Using {len(hvg)} HVGs')

    if not adataOut:
        del adata_hvg
        return hvg.tolist()
    else:
        return adata_hvg[:, hvg].copy()


def subsample(adata, batch_key, fraction=0.1, specific_batch=None, specific_cell_types=None, cell_type_key=None):
    """
        Performs Stratified subsampling on ``adata`` while keeping all samples for cell types in ``specific_cell_types``
        if passed.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        batch_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        fraction: float
            Fraction of cells out of all cells in each study to be subsampled.
        specific_cell_types: list
            if `None` will just subsample based on ``batch_key`` in a stratified way. Otherwise, will keep all samples
            with specific cell types in the list and do not subsample them.
        cell_type_key: str
            Name of the column which contains information about different cell types in ``adata.obs`` data frame.
        Returns
        -------
        adata: :class:`~anndata.AnnData`
            Subsampled annotated dataset.
    """
    studies = adata.obs[batch_key].unique().tolist()
    index = np.array([])
    if specific_cell_types and cell_type_key:
        subsampled_adata_index = adata[adata.obs[cell_type_key].isin(specific_cell_types)].obs.index
        other_adata = adata[~adata.obs[cell_type_key].isin(specific_cell_types)]
        index = np.concatenate((index, subsampled_adata_index))
    else:
        other_adata = adata

    if specific_batch:
        subsampled_adata_index = other_adata[other_adata.obs[batch_key].isin(specific_batch)].obs.index
        other_adata = other_adata[~other_adata.obs[batch_key].isin(specific_batch)]
        index = np.concatenate((index, subsampled_adata_index))

    for study in studies:
        study_index = other_adata[other_adata.obs[batch_key] == study].obs.index
        subsample_idx = np.random.choice(study_index, int(fraction * study_index.shape[0]), replace=False)
        index = np.concatenate((index, subsample_idx))

    return adata[index, :]


import scanpy as sc

ORGAN_NAMES = ['Mammary_Gland',
               'Marrow',
               'Pancreas',
               'Skin',
               'Spleen',
               'Thymus',
               'Tongue',
               'Trachea']


for organ_name in ORGAN_NAMES:
    organ = organ_name.lower()
    path = "../data/"+organ+"/"
    # path = "../../download_scripts/"+organ+"/"

    droplet = sc.read_h5ad(path+ "tabula-muris-senis-droplet-processed-official-annotations-"+organ_name+".h5ad")
    facs = sc.read_h5ad(path+ "tabula-muris-senis-facs-processed-official-annotations-"+organ_name+".h5ad")

    facs_genes = facs.var_names.values
    droplet_genes = droplet.var_names.values
    shared_genes = list(set(facs_genes).intersection(droplet_genes))

    facs = facs[:, shared_genes]
    droplet = droplet[:, shared_genes]
    droplet.obs['batch'] = 'droplet'
    facs.obs['batch'] = 'facs'

    adata = droplet.concatenate(facs)
    adata.obs['batch'] = adata.obs['batch'].cat.rename_categories({'0': 'droplet', '1': 'facs'})

    from scDI.utils import _hvg_batch
    from scDI.models.utils import train_test_split

    # taking HVGs out

    high_vgs = 4000
    adata = _hvg_batch(adata, batch_key='batch', target_genes=high_vgs, adataOut=True)
    del facs, droplet

    facs_indices = adata.obs['batch'] == 'facs'
    droplet_indices = adata.obs['batch'] == 'droplet'

    facs_train, facs_test = train_test_split(adata[facs_indices].copy())
    facs_train.write_h5ad(path+ 'facs_train.h5ad')
    facs_test.write_h5ad(path + 'facs_test.h5ad')

    droplet_train, droplet_test = train_test_split(adata[droplet_indices].copy())
    droplet_train.write_h5ad(path+ 'droplet_train.h5ad')
    droplet_test.write_h5ad(path + 'droplet_test.h5ad')

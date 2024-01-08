import torch
import numpy as np
from scDI.models.mlp import MLP
from scDI.models.trainer import Trainer
import scanpy as sc

tech = 'droplet'
organ_name = 'Thymus'
organ = organ_name.lower()
path = "../../data/" + organ + "/"
adata = sc.read_h5ad(path + tech+'_train.h5ad')

adata.X = adata.X.A
label_key = 'cell_ontology_class'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    # torch.cuda.set_device(1)
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

s_x, s_y = 1, 2
input_dim = adata.shape[1]
num_classes = np.unique(adata.obs[label_key]).shape[0]

first_l = [2048, 1024, 512]
sec_l = [512, 256, 128]
third_l = [256, 128, 64]
l2 = [0.001, 0.0001, 0]
dr_rate = [0.0, 0.025, 0.05, 0.075]

best_result = {'0-Result': 0, '1-first_l': 0, '2-sec_l': 0, '3-third_l': 0, '4-l2': 0, '5-dr_rate': 0}
results = {}
trial = 0
for f in first_l:
    for s in sec_l:
        for t in third_l:
            for l in l2:
                for d in dr_rate:

                    model = MLP(input_dim, num_classes=num_classes, layer_sizes=[f, s, t], dr_rate=d)
                    model.to(device)

                    s_x, s_y = 1, 2

                    trainer = Trainer(model, adata, label_key=label_key, s_x=s_x, s_y=s_y)
                    trainer.train_HSIC(300, 32, early_patience=30, weight_decay=l)

                    current_result = {'0-Result': trainer.accuracy,
                                      '1-first_l': f,
                                      '2-sec_l': s,
                                      '3-third_l': t,
                                      '4-l2': l,
                                      '5-dr_rate': d}
                    print('trial: ', trial, current_result)

                    if current_result['0-Result'] >= best_result['0-Result']:
                        for k in best_result.keys():
                            best_result[k] = current_result[k]
                    results[trial] = current_result
                    trial += 1

for key, value in results.items():
    print("    ", key, ": ", value)

print('Best')
for key, value in best_result.items():
    print("    ", key, ": ", value)

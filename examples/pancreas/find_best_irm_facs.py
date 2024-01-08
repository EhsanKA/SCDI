import torch
import numpy as np
from scDI.models.mlp import MLP
from scDI.models.trainer import Trainer
import scanpy as sc

tech = 'facs'
organ_name = 'Pancreas'

organ = organ_name.lower()
path = "../../data/" + organ + "/"
adata = sc.read_h5ad(path + tech+'_train.h5ad')

adata.X = adata.X.A
label_key = 'cell_ontology_class'
condition_key = 'batch'

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
reg_weight = [0.01, 0.1, 0.5, 1.5, 3, 10, 50]

best_result = {'0-Result': 0, '1-first_l': 0, '2-sec_l': 0, '3-third_l': 0, '4-reg_weight': 0, }
results = {}
trial = 0
for f in first_l:
    for s in sec_l:
        for t in third_l:
            for reg_w in reg_weight:

                model = MLP(input_dim, num_classes=num_classes, layer_sizes=[f, s, t], reg_weight=reg_w)
                model.to(device)

                trainer = Trainer(model, adata, label_key=label_key, condition_key=condition_key)
                trainer.train_IRM(300, 32, early_patience=30)

                current_result = {'0-Result': trainer.accuracy,
                                  '1-first_l': f,
                                  '2-sec_l': s,
                                  '3-third_l': t,
                                  '4-reg_weight': reg_w}
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

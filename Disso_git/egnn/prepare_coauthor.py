from torch_geometric.datasets import Coauthor
from torch_geometric.utils import to_networkx
import pickle
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import numpy as np
import sklearn.preprocessing as pp
import torch
import os

datasets = ['CS'] #,'Physics']

for dataset in datasets:
    coauth = Coauthor('data_coauthor_'+dataset,dataset)
    print("coauth done")
    data = torch.load('data_coauthor_' + dataset + '/' + dataset + '/processed/data.pt')
    print("assigned val to data")
    G = to_networkx(data[0],to_undirected=True,remove_self_loops=True)
    print("made G")
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    frc = FormanRicci(G)
    frc.compute_ricci_curvature()
    X = np.array(data[0].x)
    y_arr = np.array(data[0].y)
    Y = pp.label_binarize(y_arr, list(set(y_arr)))
    counts = [0]*15
    train_idx = []
    for i,yval in enumerate(y_arr):
        if counts[yval] < 20:
            train_idx.append(i)
            counts[yval] += 1
    test_val_idx = list(set(range(18333)).difference(set(train_idx)))
    val_idx, test_idx = test_val_idx[:500], test_val_idx[500:1500]
    os.mkdir('data_coauthor_' + dataset + '/' + dataset + '/curvatures_and_idx')
    with open('data_coauthor_' + dataset + '/' + dataset + '/curvatures_and_idx/curv_idx','wb') as f:
        pickle.dump((X,Y,train_idx,val_idx,test_idx,orc,frc),f)

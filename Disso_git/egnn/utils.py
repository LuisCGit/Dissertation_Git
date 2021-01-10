import sklearn.metrics.pairwise, scipy.sparse, pathlib, pickle, sparse
import sklearn.preprocessing as pp
import numpy as np
import networkx as nx
import awesomeml2.ops as aml_ops
import awesomeml2.graph as aml_graph
import awesomeml2.utils as aml_utils
from collections import Counter
import tensorflow as tf
import pandas as pd

def load_data(args,dataset=None):
    if dataset:
        args.data = dataset
    with open('data/'+args.data, 'rb') as f:
        X, A, Y, node_ids = pickle.load(f)
        #A = np.load("data/doubly_stoch_norm_ollivier_forman.npy")
    with open('data/'+args.data+'-tvt-'+args.data_splitting, 'rb') as f:
        idx_train, idx_val, idx_test = pickle.load(f)
    return X, Y, A, idx_train, idx_val, idx_test

def load_data_lastfm():
    with open('data_lasftm_asia/80_20', 'rb') as f:
        X,Y,train_idx,test_idx,G = pickle.load(f)
        train_idx , val_idx = train_idx[:int(len(train_idx)*0.5)], train_idx[int(len(train_idx)*0.5):]
    return X,Y,train_idx,test_idx,val_idx,G

def calc_class_weights(Y):
    nC = Y.shape[-1]
    Y = Y.reshape((-1,nC))
    N = float(Y.shape[0])
    w = []
    for i in range(nC):
        w.append( N / nC / Y[:,i].sum())
    return w

def calc_loss(y_true, logits, idx, loss_func=tf.losses.softmax_cross_entropy, W=None):

    y_true = y_true.astype(np.float32)
    logits = tf.gather(logits, idx)
    y_true = y_true[idx]
    if W is None:
        ret = loss_func(y_true, logits)
    else:
        W = tf.tensordot(y_true, W, 1)
        ret = loss_func(y_true, logits, weights=W)
    return ret


def calc_acc(y_true, y_pred, idx, is_multi_label=False):
    nC = y_true.shape[-1]
    y_true = y_true[...,idx,:].reshape((-1,nC))
    y_pred = y_pred[...,idx,:].reshape((-1,nC))
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    return acc

def load_graph(graph_path):
    """
    Reading a NetworkX graph.
    :param graph_path: Path to the edge list.
    :return graph: NetworkX object.
    """
    data = pd.read_csv(graph_path)
    edges = data.values.tolist()
    edges = [[int(edge[0]), int(edge[1])] for edge in edges]
    graph = nx.from_edgelist(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

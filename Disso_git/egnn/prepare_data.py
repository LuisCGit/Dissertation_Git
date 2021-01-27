import awesomeml2 as aml
#import awesomeml2.datasets
#import data as datasets
import utils
import awesomeml2.graph
import awesomeml2.utils
from pathlib import Path
import pickle
import sklearn.preprocessing as pp
import networkx as nx
import scipy.sparse
from scipy.sparse import csr_matrix
import numpy as np

#Packages added by me:
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

def G_from_data_file(dataset):
    file_name = "data/" + dataset
    X,A,Y,ids = np.load(file_name)
    G = nx.convert_matrix.from_scipy_sparse_matrix(A,create_using=nx.DiGraph)
    Y_idx = np.argmax(Y,axis=1)
    values_dict = {i:Y_idx[i] for i in range(2708)}
    nx.classes.function.set_node_attributes(G,values=values_dict,name='category')
    return G

def map_curvature_val(curvature,alpha=1):
    """
    maps a curvature value x from R (real line) to [0,1] via the map f(x) = e^(alpha*x)/(e^(alpha*x)+1).
    @param curvature: a curvature value (scalar)
    @param alpha: adjustable hyperparameter

    @return: returns f(curvature)
    """
    print("curvature", curvature)
    print(type(curvature))
    return np.exp(alpha*curvature)/(np.exp(alpha*curvature)+1)

def map_curvature_val_tan(curvature,alpha=1):
    """
    maps a curvature value x from R (real line) to [-1,1] via the map f(x) = arctan(x) * 2/pi.
    @param curvature: a curvature value (scalar)
    @param alpha: adjustable hyperparameter

    @return: returns f(curvature)
    """
    return np.arctan(x) * 2/np.pi


data_dir = Path('data_luis')
data_dir.mkdir(parents=True, exist_ok=True)

# for data in ['cora', 'citeseer', 'pubmed']:
#     print('data=', data)
#     #G = utils.load_data(data)
#     G = G_from_data_file("data/cora")
#     node_ids = list(G.nodes)
#
#     X = aml.graph.node_attr_matrix(G, exclude_attrs=['category'], nodes=node_ids)
#     Y = [G.nodes[i]['category'] for i in node_ids]
#     Y = pp.label_binarize(Y, list(set(Y)))
#     A = nx.adjacency_matrix(G, nodelist=node_ids)
#     print("Ummmmm bosso")
#     print("A Type:")
#     print(type(A))
#     ########################################### TRYING TO REPLACE A WITH RICCI CURVATURE ###########################################
#     print("here scparse")
#     E_norm = csr_matrix(A.shape).toarray()
#     orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
#     orc.compute_ricci_curvature()
    # frc = FormanRicci(G)
    # frc.compute_ricci_curvature()
    #
    # A = E_norm
    # ###########################################---###########################################
    # Y = aml.utils.as_numpy_array(Y)
    # with open(data_dir/data, 'wb') as f:
    #     pickle.dump((X,A,Y,node_ids), f)
    # node_id2idx = dict(zip(node_ids, range(len(node_ids))))
    # for splitting in ['602020', '051580']:
    #     id_train, id_val, id_test = utils.load_dataset_tvt(
    #         data, 'cvpr2019-'+splitting)
    #     idx_train = [node_id2idx[i] for i in id_train]
    #     idx_val = [node_id2idx[i] for i in id_val]
    #     idx_test = [node_id2idx[i] for i in id_test]
    #     with open(data_dir/(data+'-tvt-'+splitting), 'wb') as f:
    #         pickle.dump((idx_train, idx_val, idx_test), f)

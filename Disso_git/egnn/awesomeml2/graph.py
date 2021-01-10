# -*- coding: utf-8 -*-
"""
Graph/network data manipulation.
"""
import scipy.sparse
import numpy as np
import networkx as nx
import collections.abc
import itertools
import sparse

from . import ops as _ops

def normalize_adj(A, method='sym', *, axis1=-2, axis2=-1,
                  assume_symmetric_input=False,
                  check_symmetry=False, eps=1e-10,
                  array_mode=None,
                  array_default_mode='numpy',
                  array_homo_mode=None):
    """Normalize adjacency matrix defined by axis1 and axis2 in an array
    """
    print("USING NORMALIZE ADJ")
    dtype = A.dtype if np.issubdtype(A.dtype, np.floating) else np.float

    if method in ['row', 'col', 'column']:
        axis_to_sum = axis2 if method == 'row' else axis1
        norm = np.sum(A, axis_to_sum, dtype=dtype, keepdims=True)
        norm[norm==0] = eps
        norm = 1.0 / norm
        return A * norm
    elif method in ['ds', 'dsm', 'doubly_stochastic']:
        # step 1: row normalize
        print("IN THIS BLOCK")
        norm = np.sum(A, axis2, dtype=dtype, keepdims=True)
        norm[norm==0] = eps
        norm = 1.0 / norm
        P = A * norm

        # step 2: P @ P^T / column_norm
        P = _ops.swapaxes(P, axis2, -1)
        P = _ops.swapaxes(P, axis1, -2)
        norm = np.sum(P, axis=-2, dtype=dtype, keepdims=True)
        norm[norm==0] = eps
        norm = 1.0 / norm
        PT = _ops.swapaxes(P, -1, -2)
        P = np.multiply(P, norm)
        T = np.matmul(P, PT)
        T = _ops.swapaxes(T, axis1, -2)
        T = _ops.swapaxes(T, axis2, -1)
        return T
    else:
        assert method in ['sym', 'symmetric']
        treat_A_as_sym = False
        if assume_symmetric_input:
            if check_symmetry:
                _utils.assert_is_symmetric(A, axis1, axis2)
            treat_A_as_sym = True
        else:
            if check_symmetry:
                treat_A_as_sym = _utils.is_symmetric(A, axis1, axis2)

        norm1 = np.sqrt(np.sum(A, axis2, dtype=dtype, keepdims=True))
        norm1[norm1==0] = 1e-10
        norm1 = 1.0 / norm1
        if treat_A_as_sym:
            norm2 = _ops.swapaxes(norm1, axis1, axis2)
        else:
            norm2 = np.sqrt(np.sum(A, axis1, dtype=dtype, keepdims=True))
            norm2[norm2==0] = 1e-10
            norm2 = 1.0 / norm2
        return A * norm1 * norm2


def node_attr_ids(G, nodes=None):
    """Collecting the set of attribute ids associated with nodes

    Args:

      G (NetworkX Graph): Input Graph

      nodes (Iterable, NoneType): Nodes whose attribute ids will be
        collected. If it is None, collect attribute ids from all nodes
        of the graph. Default to :obj:`None`.

    Return:

      Set: The attribute id set of nodes

    Examples:

    >>> G = nx.Graph()
    >>> G.add_node(0, a=1)
    >>> G.add_node(1, b=2)
    >>> G.add_node(2, c=3)
    >>> sorted(node_attr_ids(G))
    ['a', 'b', 'c']
    >>> sorted(node_attr_ids(G, [0,1]))
    ['a', 'b']
    >>> node_attr_ids(G, [0])
    {'a'}

    """
    attr_ids = set()
    view = G.nodes(data=True)
    if nodes is None:
        nodes = G.nodes
    for i in nodes:
        attr_ids.update(view[i].keys())
    return attr_ids



def _validate_args_node_attr_func(G,
                                  include_attrs=None,
                                  exclude_attrs=None,
                                  nodes=None):
    if nodes is None:
        nodes = G.nodes
    elif not all(node in G for node in nodes):
        raise ValueError('Some nodes are not in the graph G.')


    if include_attrs is not None and exclude_attrs is not None:
        raise ValueError('include_attrs and exclude_attr cannot be specified'
                         ' together.')
    all_attrs = node_attr_ids(G)
    if include_attrs is None:
        attrs = all_attrs
    else:
        # check if attributes are in the graph
        attrs = include_attrs
        for a in attrs:
            if a not in all_attrs:
                raise KeyError('Graph has no attribute:', a)
    #attrs = node_attr_ids(G) if include_attrs is None else include_attrs
    if exclude_attrs is not None:
        exclude_attrs = set(exclude_attrs)
        attrs = [x for x in attrs if x not in exclude_attrs]

    return attrs, nodes



def node_attr_matrix(G, attrs=None, exclude_attrs=None, nodes=None):
    """Return node attributes as a scipy.sparse.coo_matrix
    """
    attrs, nodes = _validate_args_node_attr_func(
        G, attrs, exclude_attrs, nodes)
    M, N = len(nodes), len(attrs)
    # collect and check attribute shape
    attr_shapes = dict(zip(attrs, [None] * N))
    for node in G.nodes:
        for attr in G.nodes[node]:
            s = np.shape(G.nodes[node][attr])
            if attr in attr_shapes:
                if attr_shapes[attr] is None:
                    attr_shapes[attr] = s
                elif attr_shapes[attr] != s:
                    raise ValueError('Shape of attribute:', attr, ' is not'
                                     'consistent.')
    attr_shapes = [attr_shapes[x] for x in attrs]

    # collect data
    data, I, J = [], [], []
    for node_idx, node in enumerate(nodes):
        node_data = G.nodes[node]
        for attr_idx, attr in enumerate(attrs):
            try:
                val = node_data[attr]
            except KeyError:
                continue
            data.append(val)
            I.append(node_idx)
            J.append(attr_idx)
    # construct array
    return sparse.COO((I,J), data, shape=(M,N))

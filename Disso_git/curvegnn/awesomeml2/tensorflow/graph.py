# -*- coding: utf-8 -*-
"""
Graph (network) data processing using TensorFlow.
"""
import numpy as np
import tensorflow as tf

from . import ops as _tf_ops
#from . import linalg as _tf_linalg
from . import utils as _tf_utils

def normalize_adj(A, method='sym', *, axis1=-2, axis2=-1, eps=0.0,
                  assume_symmetric_input=False):
    """Normalize adjacency matrix defined by axis0 and axis1 in a tensor or sparse tensor

    Args:

      A (any-tensor): Input adjacency matrix or matrices.

      method (str): Normalization method, could be:

        * 'sym', 'symmetric': Symmetric normalization, i.e. A' =
          D^-0.5 * A * D^-0.5

        * 'row': Row normalizatiion, i.e. A' = D^-1 * A

        * 'col', 'column': Column normalization, i.e. A' = A * D^-1

      axis1 (int): Specify the first axis of the adjacency matrices. Note
        that the input A could be a batch of matrices.

      axis2 (int): Specify the second axis of the adjacency matrices.

      eps (float): Regularization small value to avoid dividing by
        zero. Default to 0.0.

      assume_symmetric_input (bool): Whether assume the input
        adjacency matrices are symmetric or not. It affects results of
        symmetric normalization only. When it is True, it will reuse
        the row sum as col sum, which will avoid the computation of
        column sum. Will need to be set as False when the inputs is
        not symmetric, otherwise the result will be incorrect. Default
        to True.

    Returns:

      any-tensor: Normalized adjacency matrix
    """
    A = tf.convert_to_tensor_or_sparse_tensor(A)
    ndims = A.get_shape().ndims
    print("USING NORMALIZE ADJ")

    if not A.dtype.is_floating:
        A = tf.cast(A, tf.float32)

    if method in ['row', 'col', 'column']:
        axis_to_sum = axis2 if method == 'row' else axis1
        norm =  tf.reduce_sum(A, axis_to_sum, keepdims=True)
        norm = 1.0 / (norm + eps)
        res = A * norm
    elif method in ['sym', 'symmetric']:
        norm1 = tf.reduce_sum(A, axis=axis2, keepdims=True)
        norm1 = 1.0 / (tf.sqrt(norm1) + eps)

        if assume_symmetric_input:
            norm2 = _tf_ops.swapaxes(norm1, axis1, axis2)
        else:
            norm2 = tf.reduce_sum(A, axis=axis1, keepdims=True)
            norm2 = 1.0 / (tf.sqrt(norm2) + eps)
        res = A * norm1 * norm2
    else:
        assert method in ['dsm', 'ds', 'doubly_stochastic']

        # step 1: row normalize
        norm = tf.reduce_sum(A, axis=axis2, keepdims=True)
        norm = 1.0 / (norm + eps)
        P = A * norm

        # step 2: P @ P^T / column_sum
        P = _tf_ops.swapaxes(P, axis2, -1)
        P = _tf_ops.swapaxes(P, axis1, -2)
        norm = tf.reduce_sum(P, axis=-2, keepdims=True)
        norm = 1.0 / (norm + eps)
        PT = _tf_ops.swapaxes(P, -1, -2)
        P = P * norm
        T = tf.matmul(P, PT)
        T = _tf_ops.swapaxes(T, axis1, -2)
        T = _tf_ops.swapaxes(T, axis2, -1)
        res = T
    return res

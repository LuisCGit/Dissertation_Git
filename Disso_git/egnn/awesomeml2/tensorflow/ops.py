# -*- coding: utf-8 -*-
"""
TensorFlow tensor and sparse tensor ops
"""
import scipy.sparse
import numpy as np
import tensorflow as tf

#from .. import ops as _ops
#from .. import config as _config
#from .. import utils as _utils
from .. import utils as _utils

#from . import utils as _tf_utils


def swapaxes(a, axis1, axis2):
    """Interchange two axes of a Tensor

    Args:

      a: input Tensor

      axis1: int, first axis

      axis2: int, second axis

    Returns:

    A tensor with axes swapped

    Examples:

    >>> import tensorflow as tf
    >>> import awesomeml.utils as aml_utils
    >>> sess = tf.Session()
    >>> a = np.random.random((2,3,4,5))
    >>> b = np.swapaxes(a, axis1=1, axis2=2)
    >>> b2 = sess.run(swapaxes(a, axis1=1, axis2=2))
    >>> np.testing.assert_array_almost_equal(b, b2)

    >>> import sparse
    >>> a = np.random.random((3,4))
    >>> b = np.swapaxes(a, axis1=0, axis2=1)
    >>> b2 = swapaxes(sparse.as_coo(a), axis1=0, axis2=1)
    >>> isinstance(b2, tf.SparseTensor)
    True
    >>> b2 = sess.run(b2)
    >>> b2 = aml_utils.as_numpy_array(b2)
    >>> np.testing.assert_array_almost_equal(b, b2)
    """
    print("USING SWAP AX")
    a = tf.convert_tensor_or_sparse_tensor(a)
    ndim = a.get_shape().ndims
    if ndim is None: # pragma: no cover
        raise ValueError('can not swapaxes for tensors with unknown shape')
    axis1 = ndim+axis1 if axis1<0 else axis1
    axis2 = ndim+axis2 if axis2<0 else axis2


    if axis1 == axis2:
        return a

    perm = list(range(ndim))
    perm[axis1] = axis2
    perm[axis2] = axis1
    if isinstance(a, tf.Tensor):
        return tf.transpose(a, perm)
    else:
        return tf.sparse_transpose(a, perm)


def moveaxis(a, axis_src, axis_dst):
    """Move an axis of a tensor to new position, similar to np.moveaxis

    Other axes remain in the original order

    Args:

      a (Tensor): the tensor whose axes should be reordered

      axis_src (int, Seq[int]): Original positions of the axes to
    move. These must be unique.

      axis_dst (int, Seq[int]): Destination position for each of the
      origianl axes. These must also be unique.

    Examples:

    >>> a = np.zeros((3, 4, 5))
    >>> moveaxis(a, 0, -1).get_shape().as_list()
    [4, 5, 3]
    >>> moveaxis(a, -1, 0).get_shape().as_list()
    [5, 3, 4]
    >>> moveaxis(a, [0, 1], [-1, -2]).get_shape().as_list()
    [5, 4, 3]
    >>> moveaxis(a, [0, 1, 2], [-1, -2, -3]).get_shape().as_list()
    [5, 4, 3]
    >>> moveaxis(a, [0, 1], [-1, -2, -3])
    Traceback (most recent call last):
      ...
    ValueError: ...
    >>> sa = scipy.sparse.random(3, 4)
    >>> moveaxis(sa, [0, 1], [-1, -2]).get_shape().as_list()
    [4, 3]
    """
    a = tf.convert_to_tensor_or_sparse_tensor(a)
#    a = _tf_utils.as_tensor_or_sparse_tensor(a)
    ndims = a.get_shape().ndims
    # src = _utils.validate_axis(
    #     axis_src, ndims, 'axis_src', accept_none=False,
    #     scalar_to_seq=True)
    # dst = _utils.validate_axis(
    #     axis_dst, ndims, 'axis_dst', accept_none=False,
    #     scalar_to_seq=True)
    src = _utils.validate_axis2(axis_src, ndims)
    dst = _utils.validate_axis2(axis_dst, ndims)

    if len(src) != len(dst):
        raise ValueError('`axis_src` and `axis_dst` arguments must have the '
                         'same number of elements')
    order = [i for i in range(ndims) if i not in src]
    for dst_1, src_1 in sorted(zip(dst, src)):
        order.insert(dst_1, src_1)
    if isinstance(a, tf.Tensor):
        res = tf.transpose(a, order)
    else:
        res = tf.sparse_transpose(a, order)
    return res

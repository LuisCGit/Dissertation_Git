# -*- coding: utf-8 -*-
"""
TensorFlow neural network layers.
"""
import tensorflow as tf
from .. import utils as _utils
from . import ops as _tf_ops
from . import utils as _tf_utils
from . import nn as _tf_nn
from . import graph as _tf_graph


class Layer(tf.layers.Layer):
    """A base layer which handles initializer, regularizer, constraints etc

    Args:

      kernel_initializer: An initializer for kernel weights. Default is None.

      bias_initializer: An initializer for bias. Default is zeros_initializer

      kernel_regularizer: An regularizer for kernel weights. Default is None.

      bias_regularizer: An regularizer for bias. Default is None.

      kernel_constraint: A projection function to be applied to the
        kernel weights after being updated by an Optimizer (e.g. used
        to implement norm constraints or value constraints for layer
        weights). The function must take as input the unprojected
        variable and must return the projected variable (which must
        have the same shape). Constraints are not safe to use when
        doing asynchronous distributed training.

      bias_constraint: A projection function to be applied to the bias
        after being updated by an Optimizer.

    """
    def __init__(self,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer
        self.kernel_constraint=kernel_constraint
        self.bias_constraint=bias_constraint

    def add_kernel(self, *args, **kwargs):
        """A warpper of self.add_variable that use self.kernel_initializer, etc as default
        """
        keys = ['initializer', 'regularizer', 'constraint']
        for key in keys:
            if key in kwargs:
                raise KeyError('argument '+key+' specified for add_kernel '
                               ' which is asssumed to use self.'+key+' . '
                               'Consider use add_variable if you want to '
                               'specify '+key+'.')
        return self.add_variable(*args, **kwargs,
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

    def add_bias(self, *args, **kwargs):
        """A wrapper of self.add_variable that use self.bias_initializer etc as default
        """
        keys = ['initializer', 'regularizer', 'constraint']
        print("before line 74")
        for key in keys:
            if key in kwargs:
                raise KeyError('argument '+key+' specified for add_kernel '
                               ' which is asssumed to use self.'+key+' . '
                               'Consider use add_variable if you want to '
                               'specify '+key+'.')
        return self.add_variable(*args, **kwargs,
                                 initializer=self.bias_initializer,
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)


class CompositeLayer(Layer):
    """Base class for layers that are build upon other layers
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # avoid conflict
        assert not hasattr(self, '_children')
        self._children = dict()

    def add_layer(self, name, layer_class, *args, **kwargs):
        """Construct a layer object and add it as sublayer

        Construct an instance of class `layer_class` and add it into
        the sublayer sets. If `layer_class` is a subclass of `Layer`,
        it will also use self.kernel_initializer, etc as default
        argument values for corresponding arguments of the
        constructor.

        Args:

          name (str): Name of the layer, also used as an ID to index
          as sublayer.

        """
        if not isinstance(name, str):
            raise ValueError('Argument `name` should be a str')
        if name in self._layers:
            raise ValueError('Layer with name '+name+' already exists.')
        if not issubclass(layer_class, tf.keras.layers.Layer):
            raise ValueError('The second argument must be a layer.')
        if issubclass(layer_class, Layer):
            # use self's initializer, regularizer and constraints as default
            kwargs2 = dict(kernel_initializer=self.kernel_initializer,
                           bias_initializer=self.bias_initializer,
                           kernel_regularizer=self.kernel_regularizer,
                           bias_regularizer=self.bias_regularizer,
                           kernel_constraint=self.kernel_constraint,
                           bias_constraint=self.bias_constraint)
            kwargs2.update(kwargs)
            kwargs = kwargs2
        layer = layer_class(*args, **kwargs)
        self._children[name] = layer
        return layer


class BiasAdd(tf.layers.Layer):
    """A layer which add bias vectors to a specific axis

    Args:
      axis (int): axis where bias will be applied to.
    """
    def __init__(self,
                 axis=-1,
                 initializer=tf.zeros_initializer(),
                 regularizer=None,
                 constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.initializer=initializer
        self.regularizer=regularizer
        self.constraint=constraint
        # self.axis=_utils.validate_axis(axis, accept_none=False,
        #                                scalar_to_seq=True)
        self.axis = _utils.validate_axis2(axis)

    def build(self, input_shape):
        ndims = len(input_shape)
        bias_shape = [input_shape[i] for i in self.axis]
        self.bias = self.add_variable('bias', bias_shape,
                                      initializer=self.initializer,
                                      regularizer=self.regularizer,
                                      constraint=self.constraint)
        super().build(input_shape)

    def call(self, input):
        print("here in line 162")
        return _tf_nn.bias_add(input, self.bias, self.axis)

def bias_add(input, *args, **kwargs):
    """Functional version of BiasAdd layer
    """
    print("here in line 168")
    print("*args,", *args)
    print("*kwargs,", **kwargs)
    layer = BiasAdd(*args, **kwargs)
    return layer.apply(input)



class GraphConv(CompositeLayer):
    """Graph Convoluton (GCN) layer
    """
    def __init__(self,
                 filters,
                 activation=None,
                 data_format='channels_last',
                 node_dropout=0.6,
                 edge_dropout=0.6,
                 multi_edge_aggregation='concat',
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.activation = activation
        self.data_format = data_format
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        self.multi_edge_aggregation = multi_edge_aggregation


    def build(self, input_shapes):
        # dropout layers
        self.node_dropout_layer = self.add_layer(
            'node_dropout', tf.layers.Dropout, self.node_dropout)
        self.edge_dropout_layer = self.add_layer(
            'edge_dropout', tf.layers.Dropout, self.edge_dropout)
        self.embeded_node_dropout_layer = self.add_layer(
            'embeded_node_dropout', tf.layers.Dropout, self.node_dropout)

        # embedding layer
        self.embedding_layer = self.add_layer(
            'embedding_layer', tf.layers.Dense, self.filters)
        self.embedding_layer2 = self.add_layer(
            'embedding_layer2', tf.layers.Dense, self.filters)

        # bias layer
        self.bias_add_layer = self.add_layer(
            'bias_add', BiasAdd,
            axis=-1 if self.data_format=='channels_last' else -2,
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)

        super().build(input_shapes)


    def call(self, inputs, training=False):
        """Apply the layer on inputs

        Args:

          inputs (Seq[any_tensor_like]): (nodes, edges) representation
          of input graph.

          training (tensor_like): Either a Python boolean, or a
          TensorFlow boolean scalar tensor (e.g. a
          placeholder). Whether to return the output in training mode
          (apply dropout) or in inference mode (return the input
          untouched).

        Return:

          Seq[any_tensor_or_seq_like]: (nodes, edges)
          representation of output graph

        """
        X, E0, E1, alpha  = inputs
        X = tf.convert_to_tensor(X)
        ec_axis = -1 if self.data_format == 'channels_last' else -3

        #E = tf.convert_to_tensor(E)
        #--------ADDED FOR PROJECT PROTO (PP)
        E0, E1 = tf.convert_to_tensor(E0), tf.convert_to_tensor(E1)
        #--------

        IC = X.get_shape().as_list()[-1]

        #out_E = E
        #--------ADDED FOR PP
        #out_E = (E0, E1)
        #--------

        # node dropout
        XD = self.node_dropout_layer(X, training=training)

        # node feature embedding
        XDW = self.embedding_layer(XD)

        # embeded node dropout
        XDWD = self.embeded_node_dropout_layer(XDW, training=training)
        print("XDWD shape", XDWD.shape)

        # edge dropout
        #ED = self.edge_dropout_layer(E, training=training)
        #--------ADDED FOR PP
        ED0, ED1 = self.edge_dropout_layer(E0) , self.edge_dropout_layer(E1)
        #--------



        # node aggregation
        (sheet1, sheet2, E0, E1) = self.node_aggregate(XDW, XDWD, ED0, ED1, E0, E1, alpha, training) #-------- replaced 'ED' w ED0,ED1 and 'out_E' w E0, E1
        #Y = tf.stack([sheet1, sheet2], axis = 2)
        out_E = tf.stack([E0, E1], axis = 2)
        print("out_E shape here butchass", out_E.shape)

        # multi-edge aggregation
        if self.multi_edge_aggregation == 'concat':
            #Y = tf.concat(tf.unstack(Y, axis=-3), -1)
            Y = tf.concat([sheet1, sheet2], axis = 1)
            print("Y shape here in concat", Y.shape)
        else:
            #Y = tf.reduce_mean(Y, -3)
            Y = (sheet1 + sheet2)/2
            print("Y shape here in mean", Y.shape)

        # add bias
        Y = self.bias_add_layer(Y)

        # activation
        if self.activation:
            Y = self.activation(Y)

        return (Y, E0,E1)


    def node_aggregate(self, XDW, XDWD, ED0, ED1, E0, E1, alpha, training):  #-------- replaced 'ED' w ED0,ED1 and 'out_E' w E0, E1
        if len(E0.shape)+1 == XDW.get_shape().ndims: #--------  changed for PP if E.get_shape().ndims == XDW.get_shape().ndims:
            print("we expanding")
            #ED = tf.expand_dims(ED, -3)
            #--------ADDED FOR PP
            ED0, ED1 = tf.expand_dims(ED0,-3) , tf.expand_dims(ED1,-3)
            #--------
        else:
            print("we move")
            #ED = _tf_ops.moveaxis(ED, -1, -3)
            #--------ADDED FOR PP
            ED0, ED1 = _tf_ops.moveaxis(ED0, -1, -3) , _tf_ops.moveaxis(ED1, -1, -3)
            #--------

        EC = 2 #--------Changed FOR PP EC = ED.get_shape().as_list()[-3]
        #ED_transformed = self.embedding_layer2(ED)
        #--------ADDED FOR PP
        ED0_transformed, ED1_transformed = self.embedding_layer2(ED0) , self.embedding_layer2(ED1)
        print("ED0_transformed shape", ED0_transformed.shape)
        #--------
        XDWD = tf.stack([XDWD]*EC, axis=-3)
        print("XDWD shape", XDWD.shape)
        #Y = ED_transformed + ED @ XDWD
        print("ED0 shape", ED0.shape)
        print("ED0@XDWD[0,:,:] shape", (ED0@XDWD[0,:,:]).shape)
        #--------ADDED FOR PP
        #tried values for alpha and max val accuracy (over at most 250 epochs)
        # {0:91.6%, 0.075:91.2%, 0.15: 91.4%}
        #tried values for alpha (and 1-alpha) and max val accuracy (over at most 250 epochs)
        # {0.15: 91.4%, 0.15(double) 0.3:90.6, 0.3(double reg): 91.2}
        sheet1 = alpha*ED0_transformed + (1-alpha)*(ED0@XDWD[0,:,:])
        sheet2 = alpha*ED1_transformed + (1-alpha)*(ED1@XDWD[1,:,:])
        #--------
        print("sheet1 shape", sheet1.shape)
        return (sheet1, sheet2, E0, E1)



def graph_conv(inputs, filters, training=False, **kwargs):
    layer = GraphConv(filters, **kwargs)
    return layer.apply(inputs, training=training)


class GraphAttention(GraphConv):
    """Graph attention (GAT) layer
    """
    def __init__(self, *args,
                 edge_normalize='row',
                 beta=1.0,
                 adaptive=True,
                 eps=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_normalize = edge_normalize
        self.beta = beta
        self.adaptive = adaptive
        self.eps = eps

    def build(self, input_shapes):
        self.linear1 = self.add_layer('linear1', tf.layers.Dense, 1, use_bias=False)
        self.linear2 = self.add_layer('linear2', tf.layers.Dense, 1, use_bias=True)
        super().build(input_shapes)


    def node_aggregate(self, XDW, XDWD, E, ED, out_E, training):
        # normalize shape of XDWD and ED
        ndims_E = E.get_shape().ndims
        if ndims_E == XDW.get_shape().ndims:
            E = tf.expand_dims(E, -1)

        f_1 = tf.expand_dims(self.linear1(XDW), -2)
        f_2 = tf.expand_dims(self.linear2(XDW), -3)
        logits = tf.exp(tf.nn.leaky_relu(f_1 + f_2)) * E
        coef = _tf_graph.normalize_adj(logits, self.edge_normalize,
                                   axis1=-3, axis2=-2,
                                   assume_symmetric_input=False,
                                   eps=self.eps)
        if self.beta != 1.0:
            coef = coef * self.beta + E * (1-self.beta)
        if self.adaptive:
            out_E = coef
        coef = self.edge_dropout_layer(coef, training=training)
        coef = _tf_ops.moveaxis(coef, -1, -3)
        #XDWD = tf.expand_dims(XDWD, -3)
        EC = coef.get_shape().as_list()[-3]
        XDWD = tf.stack([XDWD]*EC, axis=-3)
        Y = coef @ XDWD
        return Y, out_E



def graph_attention(inputs, filters, training=False, *args,  **kwargs):
    """Functional wrapper of :class:`GraphAttend`

    Args:

      inputs (Seq[any_tensor_like]): (nodes, edges) representation of
    input graph

      filters (int): Number of filters (output dimensions) of result
      node features

      training (bool, tensor): whether it is in training mode

    """
    layer = GraphAttention(filters, **kwargs)
    return layer.apply(inputs, training=training)

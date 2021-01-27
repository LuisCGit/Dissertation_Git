import time,math,datetime,argparse,random,scipy.sparse,csv
import sklearn,sklearn.metrics,sklearn.cluster
import tensorflow as tf
import awesomeml2 as aml
import awesomeml2.utils as aml_utils
#import awesomeml2.tensorflow.layers as aml_layers
import awesomeml2.tensorflow.layers_neww_parallel as aml_layers

import awesomeml2.tensorflow.ops as aml_tf_ops
import awesomeml2.graph as aml_graph
import utils
from pathlib import Path
import sklearn.preprocessing as pp
import numpy as np
import sparse
import pickle

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#Packages added by me:
from prepare_data import G_from_data_file, map_curvature_val
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from scipy.sparse import csr_matrix
from itertools import product

# ************************************************************
# args
# ************************************************************
parser = argparse.ArgumentParser()
# general
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--no-test', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight-decay', type=float, default=5e-4)

# data
parser.add_argument('--data', type=str, default='cora')
parser.add_argument('--data-splitting', type=str, default='602020')

# model
parser.add_argument('--layer-type', type=str, default='gcn')
parser.add_argument('--encode-edge-direction', action='store_true')
parser.add_argument('--edge-norm', type=str, default='row')
parser.add_argument('--adaptive', action='store_true', default=False)
parser.add_argument('--weighted', action='store_true', default=False)
parser.add_argument('--alpha', type=float, default=0)

alpha_vals = np.linspace(0,0.5,11)
curvature_mapping_alpha = [1.0,4.0]
normalization = ['dsm','row']

test_accs = np.zeros((len(alpha_vals),len(curvature_mapping_alpha),len(normalization)))

def enumerated_product(*args):
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))

if __name__ == '__main__' and '__file__' in globals():
    args = parser.parse_args()
else:
    args = parser.parse_args([])

for idx, param_tup in enumerated_product(alpha_vals, curvature_mapping_alpha,normalization):
        alpha_val,curv_mapping_alpha,norm = param_tup
        args.alpha = alpha_val
        args.edge_norm = norm
        print("params", param_tup)


        # ************************************************************
        # load data
        # ************************************************************
        A = np.random.uniform(size=(18333,18333))
        with open('data_coauthor_' + 'CS' + '/' + 'CS' + '/curvatures_and_idx/curv_idx','rb') as f:
            X,Y,idx_train,idx_val,idx_test,orc,frc = pickle.load(f)

        K = A.shape[1] if X is None else X.shape[0]
        nC = Y.shape[1]
        W = None
        if args.weighted:
            W = utils.calc_class_weights(Y[...,idx_train,:])

        # ************************************************************
        # calculate node features
        # ************************************************************
        vals = []
        X = X.astype(np.float32)
        EYE = scipy.sparse.eye(K, dtype=np.float32, format='coo')
        A = A.astype(np.float32)

        # normalized x
        rowsum = sparse.as_coo(X).sum(axis=-1, keepdims=True)
        #rowsum.data[rowsum.data==0] = 1e-10
        rowsum.data = 1.0 / rowsum.data
        vals.append((sparse.as_coo(X) * rowsum).to_scipy_sparse())
        nodes = scipy.sparse.hstack(vals)
        nodes = nodes.toarray()

        # ************************************************************
        # calculate edge features
        # ************************************************************
        vals = []
        EYE = scipy.sparse.eye(K, dtype=np.float32, format='coo')


        ########################################### TRYING TO REPLACE A WITH RICCI CURVATURE ###########################################
        ollivier_curv_vals, forman_curv_vals = csr_matrix(A.shape).toarray(), csr_matrix(A.shape).toarray()

        for tup in orc.G.edges:
            i,j = tup[0], tup[1]
            ollivier_curv_vals[i][j] = map_curvature_val(orc.G[i][j]['ricciCurvature'],alpha = curv_mapping_alpha)
            forman_curv_vals[i][j] = map_curvature_val(frc.G[i][j]['formanCurvature'], alpha = curv_mapping_alpha)

        #edge_feat_list = [ollivier_curv_vals]
        edge_feat_list = [ollivier_curv_vals, forman_curv_vals]
        ########################################### END ###########################################
        for mat in edge_feat_list:
            vals.append((mat+mat.transpose()+EYE>0).astype(np.float32))
            if args.encode_edge_direction:
                vals.append((mat+EYE>0).astype(np.float32))
                vals.append((mat.transpose()+EYE>0).astype(np.float32))
        vals = [sparse.as_coo(x) for x in vals]
        vals = sparse.stack(vals, axis=0)
        vals = vals.todense()

        vals = aml_graph.normalize_adj(vals, args.edge_norm, assume_symmetric_input=False)
        temp_vals = vals
        vals = [vals]
        #ret = sparse.concatenate(vals, 1)
        ret = np.concatenate(vals, 1)


        edges = np.transpose(ret, [1,2,0])
        #edges = edges.todense()

        # ************************************************************
        # construct model
        # ************************************************************
        def layer(layer_type, inputs, dim, training, args, **kwargs):
            """A wrapper to dispatch different layer construction
            """
            if layer_type.lower() == 'gcn':
                return aml_layers.graph_conv(
                    inputs, dim, training,
                    **kwargs)
            elif layer_type.lower() == 'gat':
                return aml_layers.graph_attention(
                    inputs, dim, training,
                    eps=1e-10,
                    # we have a bug in the code for 'dsm' and 'sym' of gat
                    #edge_normalize=args.edge_norm,
                    adaptive=args.adaptive,
                    **kwargs)
            else:
                raise ValueError('layer type:', layer_type, ' not supported.')

        # reset computing graph
        tf.reset_default_graph()

        # input layer
        training = tf.placeholder(dtype=tf.bool, shape=())
        h, E0, E1 = nodes, edges[:,:,0], edges[:,:,1]

        # hidden layers
        h, E0, E1 = layer(args.layer_type, (h, E0, E1, args.alpha), 64, training, args, activation=tf.nn.elu)
        #--------ADDED FOR PP
        # sheet1, sheet2, E0, E1 = layer(args.layer_type, (h, edges), 64, training, args, activation=tf.nn.elu)
        #
        # h = tf.concat([sheet1, sheet2], axis = 0)
        # edges = tf.concat([E0, E1], axis = 0)
        #--------
        print("h shape", h.shape)
        print("edges shape", edges.shape)


        # classification layer
        logits,_,_ = layer(args.layer_type, (h, E0, E1, args.alpha), nC, training, args,
                         multi_edge_aggregation='mean')
        Yhat = tf.one_hot(tf.argmax(logits, axis=-1), nC)
        loss_train = utils.calc_loss(Y, logits, idx_train, W=W)
        loss_val = utils.calc_loss(Y, logits, idx_val)
        loss_test = utils.calc_loss(Y, logits, idx_test)

        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if
                           'bias' not in v.name and
                           'gamma' not in v.name]) * args.weight_decay
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(loss_train + lossL2)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # ************************************************************
        # training
        # ************************************************************
        ckpt_dir = Path('./ckpt')
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir/'checkpoint.ckpt'
        print('ckpt_path=', ckpt_path)

        bad_epochs = 0
        loss_stop = math.inf
        acc_stop = -math.inf
        saver = tf.train.Saver()
        nan_happend = False

        #edges_init = np.random.uniform(size=(18333,18333,2)).astype(np.float32)
        # edges_plhdr = tf.placeholder(dtype=tf.float32, shape=[18333, 18333,2])
        # edges_layer = tf.Variable(edges_plhdr)
        #edges_layer = tf.get_variable('edges_layer', [18333, 18333,2])
        #with tf.Session() as sess:
        #    sess.run(tf.global_variables_initializer())


        with tf.Session() as sess:
            sess.run(init_op)# feed_dict = {edges_plhdr:layer(args.layer_type, (h, edges), 64, training, args, activation=tf.nn.elu)[1]})
            #sess.run(edges_layer.assign(edges_plhdr), {edges_plhdr: edges_init})

            t0 = time.time()
            for epoch in range(args.epochs):
                t = time.time()
                # training step
                sess.run([train_op], feed_dict={training:True}) #, feed_dict={training:True,edges_plhdr:layer(args.layer_type, (h, edges), 64, training, args, activation=tf.nn.elu)[1]})

                # validation step
                [loss_train_np, loss_val_np, Yhat_np] = sess.run(
                    [loss_train, loss_val, Yhat],
                    feed_dict={training:False}) #edges_plhdr:layer(args.layer_type, (h, edges), 64, training, args, activation=tf.nn.elu)[1]})
                acc_train = utils.calc_acc(Y, Yhat_np, idx_train)
                acc_val = utils.calc_acc(Y, Yhat_np, idx_val)

                if np.isnan(loss_train_np):
                    nan_happend = True
                    print('NaN loss, stop!')
                    break

                print('Epoch=%d, loss=%.4f, acc=%.4f | val: loss=%.4f, acc=%.4f t=%.4f' %
                      (epoch, loss_train_np, acc_train, loss_val_np, acc_val, time.time()-t))
                if loss_val_np <= loss_stop:
                    bad_epochs = 0
                    if not args.no_test:
                        #saver.save(sess, str(ckpt_path))
                        pass
                    loss_stop = loss_val_np
                    acc_stop = acc_val
                else:
                    bad_epochs += 1
                    if bad_epochs == args.patience:
                        print('Early stop - loss=%.4f acc=%.4f' % (loss_stop, acc_stop))
                        print('totoal time {}'.format(
                            datetime.timedelta(seconds=time.time()-t0)))
                        break

            # evaluation step
            # load check point
            if not args.no_test or nan_happend:
                saver.restore(sess, str(ckpt_path))
                [loss_test_np, Yhat_np] = sess.run(
                    [loss_test, Yhat], feed_dict={training:False})#edges_plhdr:layer(args.layer_type, (h, edges), 64, training, args, activation=tf.nn.elu)[1]})
                acc = utils.calc_acc(Y, Yhat_np, idx_test)
                test_accs[idx] = acc
                print('Testing - loss=%.4f acc=%.4f' % (loss_test_np, acc))
                print("made to test phase")

import time,math,datetime,argparse,random,scipy.sparse,csv
import sklearn,sklearn.metrics,sklearn.cluster
import tensorflow as tf
import awesomeml2 as aml
import awesomeml2.utils as aml_utils
#import awesomeml2.tensorflow.layers_neww as aml_layers
import awesomeml2.tensorflow.layers as aml_layers

import awesomeml2.tensorflow.ops as aml_tf_ops
import awesomeml2.graph as aml_graph
import utils
from pathlib import Path
import sklearn.preprocessing as pp
import numpy as np
import sparse


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from prepare_data import G_from_data_file, map_curvature_val
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from scipy.sparse import csr_matrix
from networkx.generators.community import stochastic_block_model
import sklearn.preprocessing as pp
from sklearn.model_selection import ShuffleSplit
import pickle

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

if __name__ == '__main__' and '__file__' in globals():
    args = parser.parse_args()
else:
    args = parser.parse_args([])
print(args)


# ************************************************************
# generate SBM data
# ************************************************************
pVals = np.linspace(0.05,0.23,10)
qVals = np.linspace(0.0,0.045,10)
with open('data_sbm/SBM_dict','rb') as f:
    SBM_dict = pickle.load(f)
with open('data_sbm/SBM_dict_rest','rb') as f:
    SBM_dict_full = pickle.load(f)
with open('data_sbm/SBM_dict_lastone','rb') as f:
    SBM_dict_last = pickle.load(f)
for key in SBM_dict:
    SBM_dict_full[key] = SBM_dict[key]
for key in SBM_dict_last:
    SBM_dict_full[key] = SBM_dict_last[key]

num_trials = 10
test_accs = np.zeros((len(qVals),len(pVals),num_trials))
for trial in range(num_trials):
    print("trial #: ", trial)
    for q,p in SBM_dict_full:
        print("q, p", (q,p))
        G,orc,frc = SBM_dict_full[(q,p)]

        X = np.random.uniform(size=(len(G.nodes),10))
        A = np.random.uniform(size=(len(G.nodes),len(G.nodes)))
        Y = np.array([G.nodes[node]['block'] for node in G.nodes]).reshape((len(G.nodes),))
        Y = pp.label_binarize(Y, list(set(Y)))
        rs = ShuffleSplit(n_splits=1, test_size=.2)
        rs.get_n_splits(X)
        for train_index, test_index in rs.split(X):
            idx_train, idx_test = train_index, test_index
        idx_train, idx_val = idx_train[:len(idx_train)//2], idx_train[len(idx_train)//2:]


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
            ollivier_curv_vals[i][j] = map_curvature_val(orc.G[i][j]['ricciCurvature'])
            forman_curv_vals[i][j] = map_curvature_val(frc.G[i][j]['formanCurvature'])

        edge_feat_list = [ollivier_curv_vals,forman_curv_vals]
        #edge_feat_list = [forman_curv_vals]
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
                    adaptive=args.adaptive,
                    **kwargs)
            else:
                raise ValueError('layer type:', layer_type, ' not supported.')

        # reset computing graph
        tf.reset_default_graph()

        # input layer
        training = tf.placeholder(dtype=tf.bool, shape=())
        h, edges = nodes, edges
        # hidden layers
        h, edges = layer(args.layer_type, (h, edges), 32, training, args, activation=tf.nn.elu)

        # classification layer
        logits,_ = layer(args.layer_type, (h, edges), nC, training, args,
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
        with tf.Session() as sess:
            sess.run(init_op)

            t0 = time.time()
            for epoch in range(args.epochs):
                t = time.time()
                # training step
                sess.run([train_op], feed_dict={training:True})

                # validation step
                [loss_train_np, loss_val_np, Yhat_np] = sess.run(
                    [loss_train, loss_val, Yhat],
                    feed_dict={training:False})
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
                        saver.save(sess, str(ckpt_path))
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
                    [loss_test, Yhat], feed_dict={training:False})
                acc = utils.calc_acc(Y, Yhat_np, idx_test)
                q_idx,p_idx = np.where(qVals==q)[0][0], np.where(pVals==p)[0][0]
                test_accs[q_idx,p_idx,trial] = acc
                print('Testing - loss=%.4f acc=%.4f' % (loss_test_np, acc))

        np.save('data_sbm/heatmap_array_og_architecture',test_accs)

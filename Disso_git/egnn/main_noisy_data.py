import time,math,datetime,argparse,random,scipy.sparse,csv
import sklearn,sklearn.metrics,sklearn.cluster
import tensorflow as tf
import awesomeml2 as aml
import awesomeml2.utils as aml_utils
import awesomeml2.tensorflow.layers as aml_layers
import awesomeml2.tensorflow.layers_neww_parallel as aml_layers_p

import awesomeml2.tensorflow.ops as aml_tf_ops
import awesomeml2.graph as aml_graph
import utils
from pathlib import Path
import sklearn.preprocessing as pp
import numpy as np
import sparse

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#Packages added by me:
from prepare_data import G_from_data_file, map_curvature_val, map_curvature_val_tan
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from scipy.sparse import csr_matrix
from itertools import product
import pickle

# ************************************************************
# args
# ************************************************************
parser = argparse.ArgumentParser()
# general
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--no-test', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--num_trials', type=int, default=5)


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

#var_vals = np.logspace(-5,0,5)

#var_vals = [0.5, 1.0, 1.5, 2.0 , 3.0]
var_vals = [0, 0.2, 0.4, 1.0 , 1.5, 2.0, 3.0, 3.5, 4.0] (OG FOR PUBMED)
#var_vals = [1.0 , 1.5, 2.0, 3.0, 3.5, 4.0] (FOR NOISY2 PUBMED)
#var_vals = [3.0, 3.5, 4.0] (FOR NOISY3 PUBMED)
#var_vals = [3.5,4.0] (FOR NOISY4 PUBMED)
datasets =  ['CS']#['cora','citeseer','pubmed'] #['CS'] #['pubmedm citeseer',
alpha_vals = np.linspace(0,1,6)[:-2]

test_accs = np.zeros((len(datasets),len(alpha_vals),len(var_vals),args.num_trials,args.epochs,2)) #loss val, acc test
for d, dataset in enumerate(datasets):
    print("dataset: ", dataset)
    if dataset == 'CS':
        with open('data_coauthor_' + dataset + '/' + dataset + '/curvatures_and_idx/curv_idx','rb') as f:
            X,Y,idx_train,idx_val,idx_test,orc,frc = pickle.load(f)
            A = np.random.uniform(size=(X.shape[0],X.shape[0]))
            ollivier_curv_vals, forman_curv_vals = csr_matrix(A.shape).toarray(), csr_matrix(A.shape).toarray()

            for tup in orc.G.edges:
                i,j = tup[0], tup[1]
                ollivier_curv_vals[i][j] = map_curvature_val(orc.G[i][j]['ricciCurvature'],alpha = 4)
                forman_curv_vals[i][j] = map_curvature_val(frc.G[i][j]['formanCurvature'], alpha = 4)

    else:
        ########################################### TRYING TO REPLACE A WITH RICCI CURVATURE ###########################################
        G = G_from_data_file(dataset)
        args.data = dataset
        X, Y, A, idx_train, idx_val, idx_test = utils.load_data(args)
        ollivier_curv_vals, forman_curv_vals = csr_matrix(A.shape).toarray(), csr_matrix(A.shape).toarray()
        orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
        orc.compute_ricci_curvature()
        frc = FormanRicci(G)
        frc.compute_ricci_curvature()
        for tup in orc.G.edges:
            i,j = tup[0], tup[1]
            ollivier_curv_vals[i][j] = map_curvature_val(orc.G[i][j]['ricciCurvature'],alpha = 1)
            forman_curv_vals[i][j] = map_curvature_val(frc.G[i][j]['formanCurvature'],alpha = 1)
    K = A.shape[1] if X is None else X.shape[0]
    D = X.shape[1]
    nC = Y.shape[1]
    W = None
    if args.weighted:
        W = utils.calc_class_weights(Y[...,idx_train,:])
    # ************************************************************
    # calculate edge features
    # ************************************************************
    EYE = scipy.sparse.eye(K, dtype=np.float32, format='coo')
    edge_feat_list = [ollivier_curv_vals,forman_curv_vals]


    vals = []
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

    for v,var in enumerate(var_vals):

        # ************************************************************
        # calculate node features and add gaussian noise
        # ************************************************************
        vals = []
        X = X.astype(np.float32)
        X_noisy = X + np.random.multivariate_normal(mean = [0]*D, cov = np.identity(D)*var,size=K)
        print("shape of noise matrix")
        print(np.random.multivariate_normal(mean = [0]*D, cov = np.identity(D)*var,size=K).shape)
        X_noisy = X_noisy.astype(np.float32)
        EYE = scipy.sparse.eye(K, dtype=np.float32, format='coo')
        A = A.astype(np.float32)

        # normalized x
        rowsum = sparse.as_coo(X_noisy).sum(axis=-1, keepdims=True)
        #rowsum.data[rowsum.data==0] = 1e-10
        rowsum.data = 1.0 / rowsum.data
        vals.append((sparse.as_coo(X_noisy) * rowsum).to_scipy_sparse())
        nodes = scipy.sparse.hstack(vals)
        nodes = nodes.toarray()

        for a,alpha_val in enumerate(alpha_vals):
            print("a, v : ", (a,v))
            if v == 0:
                if a in range(3):
                    continue
            # ************************************************************
            # construct model
            # ************************************************************
            def layer(layer_type, inputs, dim, training, args, **kwargs):
                """A wrapper to dispatch different layer construction
                """
                if layer_type.lower() == 'gcn':
                        return aml_layers_p.graph_conv(
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
            training = tf.placeholder(dtype=tf.bool, shape=())

            # input layer
            h, E0, E1 = nodes, edges[:,:,0], edges[:,:,1]
            print("E0, E1 shapes", (E0.shape,E1.shape))

            # hidden layers
            h, E0, E1 = layer(args.layer_type, (h, E0, E1, alpha_val), 64, training, args, activation=tf.nn.elu)

            # classification layer
            logits,_,_ = layer(args.layer_type, (h, E0, E1, alpha_val), nC, training, args,
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
            for j in range(args.num_trials):
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
                        acc_test = utils.calc_acc(Y, Yhat_np, idx_test)

                        test_accs[d,a,v,j,epoch,0] = loss_val_np
                        test_accs[d,a,v,j,epoch,1] = acc_test

                        np.save("coauthorCS_noisy",test_accs)

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
                    # if not args.no_test or nan_happend:
                    #     saver.restore(sess, str(ckpt_path))
                    #     [loss_test_np, Yhat_np] = sess.run(
                    #         [loss_test, Yhat], feed_dict={training:False})
                    #     acc = utils.calc_acc(Y, Yhat_np, idx_test)
                    #     test_accs[i,j] = acc
                    #     np.save("curvegnn_cora_curvatures_dense",test_accs)
                    #     print('Testing - loss=%.4f acc=%.4f' % (loss_test_np, acc))

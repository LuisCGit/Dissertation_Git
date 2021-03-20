import random, argparse
import utils
import pickle
# ************************************************************
# args
# ************************************************************
parser = argparse.ArgumentParser()


# data
parser.add_argument('--data', type=str, default='cora')
parser.add_argument('--data-splitting', type=str, default='602020')

# model
if __name__ == '__main__' and '__file__' in globals():
    args = parser.parse_args()
else:
    args = parser.parse_args([])

train_vol_vals = [0.01,0.05,0.1,0.2,0.4,0.6,0.8]
idx_dicts = []
for tv,train_vol_val in enumerate(train_vol_vals):
    temp = {}
    if args.data == 'CS':
        with open('data_coauthor_' + args.data + '/' + args.data + '/curvatures_and_idx/curv_idx','rb') as f:
            X,Y,idx_train,idx_val,idx_test,orc,frc = pickle.load(f)
    else:
        X, Y, A, idx_train, idx_val, idx_test = utils.load_data(args)
    shuff = idx_train + idx_val + idx_test
    random.shuffle(shuff)
    t,v = int(train_vol_val*len(shuff)), int(len(shuff)*(1-train_vol_val)/2)
    idx_train, idx_val, idx_test = shuff[:t], shuff[t:t+v], shuff[t+v:]
    temp['idx_train'], temp['idx_val'], temp['idx_test'] = idx_train, idx_val, idx_test
    idx_dicts.append(temp)

with open('data/train_vol_idx_dicts','wb') as f:
    pickle.dump(idx_dicts,f)

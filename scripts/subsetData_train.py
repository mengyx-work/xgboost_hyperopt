import os, sys, time, random
import pandas as pd
import numpy as np
from os.path import join

dep_var_name = 'Response'
idx_col_name = 'Id'

start_time = time.time()
data_path = '/home/ymm/kaggle/bosch_data/bosch_FE_full_data_xgb'
train_file_name = 'FE_raw_train_num_dat_data.csv'

## probe the data
#idx_train =  pd.read_csv(join(data_path, train_file_name), usecols=[idx_col_name], index_col=idx_col_name)
#print 'reading the entire data taking {} seconds, the training data shape is: {}'.format(round(time.time()-start_time, 0), idx_train.shape)
#tot_row_num = int(idx_train.shape[0])
tot_row_num = 1183747
fold_num = 4
idxList = [int(1.*i/fold_num*tot_row_num) for i in range(0, fold_num+1)]

params = {}
params["eta"]                      = 0.0075
params["subsample"]                = 0.8
params["colsample_bytree"]         = 0.8
params["num_round"]                = 501
params["max_depth"]                = 5
params["gamma"]                    = 0
params["metrics"]                  = 'auc'
params['eval_metric']              = 'auc'
params["seed"]                     = 999
params['verbose_eval']             = 50
## whether to use weights
params['use_base_score']           = True
params['use_weights']              = True
#params['use_scale_pos_weight']     = True
params["val"]                      = False


## read certain section of data
for i in range(fold_num):
    start_time = time.time()
    skiprowsList = range(1, idxList[i]+1)
    skiprowsList.extend(range(idxList[i+1]+1, tot_row_num+1))
    train = pd.read_csv(join(data_path, train_file_name), usecols=[idx_col_name], index_col=idx_col_name, skiprows=skiprowsList)
    print 'for the section {}, the shape of data is {}, taking {} seconds'.format(i, train.shape, round(time.time() - start_time, 0))
    #print train.tail()


 


import pandas as pd
import numpy as np
import os, sys, time
import yaml

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils
from utils.bosch_functions import load_processed_bosch_data
#from utils.models import RandomForestModel, ExtraTreeModel
from utils.wrapped_xgboost import xgboost_classifier
from utils.validation_tools import score_MCC, grid_search_cross_validate_model, list_const_params, combine_tuning_params

dep_var_name = 'Response'
data_index= 'NaN'

'''
project_path = '/mnt/home/ymm/kaggle/compete/current'
## fully processed data
#data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'
## encoded only data
data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_processed_data' 

yaml_file = 'bosch_processed_data_dict.yml'
with open(os.path.join(project_path, yaml_file), 'r') as yml_stream:
    data_dict = yaml.load(yml_stream)

data_file = os.path.join(data_path, data_dict[data_index]['train_file'])
print 'loading data from ', data_file
train = pd.read_csv(data_file, index_col='Id')
'''

#'''
## 5 bins data
project_yml_path = '/home/ymm/kaggle/compete/current/model_6_bins_data'
data_path = '/home/ymm/kaggle/bosch_data/bosch_complete_processed_6_bins_data'
data_yaml_file = 'complete_subset_data_6_bins_dict.yml'
#'''

train = load_processed_bosch_data(data_path, data_yaml_file, data_index=data_index)

const_param_dict = {}
const_param_dict["eta"]                      = 0.001
const_param_dict["subsample"]                = 1.0
const_param_dict["colsample_bytree"]         = 1.0
const_param_dict["num_round"]                = 5000
const_param_dict["max_depth"]                = 5
const_param_dict["gamma"]                    = 0
const_param_dict["metrics"]                  = 'auc'
const_param_dict['eval_metric']              = 'auc'
const_param_dict["seed"]                     = 100
const_param_dict["use_weights"]              = True
const_param_dict["val"]                      = False

const_param_dict = list_const_params(const_param_dict)
tuning_param_dict = {'num_round' : [2500, 3500, 4500, 5500], 'max_depth' : [3, 4, 5, 6, 7]}
#tuning_param_dict = {'subsample': [1.0, 0.9, 0.8, 0.7, 0.6], 'colsample_bytree' : [1.0, 0.9, 0.8, 0.7, 0.6], 'max_depth' : [4, 5, 6], 'seed' : [0, 999]}
param_dict = combine_tuning_params(const_param_dict, tuning_param_dict)

cv_fold_num = 2
result_file_name = 'xgb_data_bin_{}_GridSearch_{}_fold_Results.csv'.format(data_index, cv_fold_num)
grid_search_cross_validate_model(train, dep_var_name, xgboost_classifier, score_MCC, param_dict, cv_fold_num, result_file= result_file_name, is_xgb_model=True)


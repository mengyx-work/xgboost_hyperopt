import pandas as pd
import numpy as np
import os, sys, time
import yaml

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils
from utils.models import RandomForestModel, ExtraTreeModel
from utils.validation_tools import score_MCC, grid_search_cross_validate_model, list_const_params

project_path = '/mnt/home/ymm/kaggle/compete/current'
data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'

yaml_file = 'bosch_processed_data_dict.yml'
with open(os.path.join(project_path, yaml_file), 'r') as yml_stream:
    data_dict = yaml.load(yml_stream)

dep_var_name = 'Response'
data_index= '0'
data_file = os.path.join(data_path, data_dict[data_index]['train_file'])
print 'loading data from ', data_file
train = pd.read_csv(data_file, index_col='Id')

const_param_dict = {'random_state' : 0}
const_param_dict = list_const_params(const_param_dict)
tuning_param_dict = {'n_estimators':range(500, 5000, 500), 'max_depth':range(4, 11)}

param_dict = const_param_dict.copy()
param_dict.update(tuning_param_dict)

grid_search_cross_validate_model(train, dep_var_name, ExtraTreeModel, score_MCC, param_dict, 3, result_file='extraTree_GridSearch_Results.csv')


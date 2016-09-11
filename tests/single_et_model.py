import pandas as pd
import numpy as np
import os, sys, time
import yaml
import cPickle as pickle

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils
from utils.models import RandomForestModel, ExtraTreeModel
from utils.validation_tools import score_MCC, create_validation_index
from utils.bosch_functions import load_processed_bosch_data

models_yaml_file = 'trained_combined_model.yml'
project_path = '/mnt/home/ymm/kaggle/compete/current/combined_model'
#project_path = '/Users/ymm/Google_Drive/Kaggle/compete/current/combined_model'

dep_var_name     = 'Response'
project_yml_path = '/mnt/home/ymm/kaggle/compete/current'
data_path        = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'
data_yaml_file   = 'bosch_processed_data_dict.yml'

train = load_processed_bosch_data(data_path, project_yml_path, data_yaml_file, data_index='0')
print train.shape
train_index,valid_index = create_validation_index(train, 0.2, dep_var_name, True)  
valid_data  = train.ix[valid_index]
train       = train.ix[train_index]
valid_label = valid_data[dep_var_name]


model_params =  {'random_state' : 0, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
model = ExtraTreeModel(model_params)
model.fit(train, dep_var_name)
valid_result = model.predict(valid_data)
print score_MCC(valid_label, valid_result)

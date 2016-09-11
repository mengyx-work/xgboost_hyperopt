import pandas as pd
import numpy as np
import os, sys, time
import yaml
import cPickle as pickle

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
import utils
from utils.models import RandomForestModel, ExtraTreeModel
from utils.validation_tools import score_MCC, grid_search_cross_validate_model, list_const_params
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
train_label = train[dep_var_name]
tmp_train = train.copy()
tmp_train.drop(dep_var_name, axis=1, inplace=True)

model_params =  {'random_state' : 0, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
model = ExtraTreeModel(model_params)
model.fit(tmp_train, train_label)
res = model.predict(tmp_train)

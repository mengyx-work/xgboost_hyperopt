import pandas as pd
import numpy as np
import os, sys, time
import yaml
import cPickle as pickle

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
from utils.bosch_functions import load_processed_bosch_data
from utils.models import train_combined_model, predict_combined_model
from utils.validation_tools import score_MCC, create_validation_index

dep_var_name = 'Response'
raw_models_yaml_file = 'raw_combined_models.yml'
trained_model_yaml_file = 'trained_combined_model.yml'
project_path = '/mnt/home/ymm/kaggle/compete/current/combined_model'
trained_model_yaml_file = 'trained_combined_model.yml'

## load training data
project_yml_path = '/mnt/home/ymm/kaggle/compete/current'
data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'
data_yaml_file = 'bosch_processed_data_dict.yml'

train = load_processed_bosch_data(data_path, project_yml_path, data_yaml_file, data_index='0')
train_index,valid_index = create_validation_index(train, 0.2, dep_var_name, True)  
valid_data = train.ix[valid_index]
train      = train.ix[train_index]

## train the comined model
train_combined_model(train, dep_var_name, raw_models_yaml_file, project_path, trained_model_yaml_file)

pred_df = predict_combined_model(valid_data, project_path, trained_model_yaml_file, score_MCC, dep_var_name)

pred_df.to_csv('tmp.csv')



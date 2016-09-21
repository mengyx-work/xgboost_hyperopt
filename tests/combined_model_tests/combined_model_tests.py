import pandas as pd
import numpy as np
import os, sys, time
import yaml
import cPickle as pickle

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
from utils.bosch_functions import load_processed_bosch_data
from utils.models import CombinedModel
from utils.validation_tools import score_MCC, MCC, create_validation_index, cross_validate_model

dep_var_name = 'Response'

## params for combined model
raw_models_yaml_file    = 'raw_combined_models.yml'
trained_model_yaml_file = 'trained_combined_model.yml'
project_path            = './tmp'
raw_models_yaml_path    = './'

## 15 bins data
project_yml_path = '/mnt/home/ymm/kaggle/compete/current'
data_path = '/home/ymm/kaggle/bosch_data/bosch_complete_processed_15_bins_data'
data_yaml_file = 'bosch_processed_data_dict.yml'

train = load_processed_bosch_data(data_path, data_yaml_file, data_index='0', nrows=10000)

## train the comined model
combined_model_params = {}
combined_model_params['raw_models_yaml_file']   = raw_models_yaml_file
combined_model_params['raw_models_yaml_path']   = raw_models_yaml_path 
combined_model_params['project_path']           = project_path
combined_model_params['models_yaml_file']       = trained_model_yaml_file
## build the combined model
combined_model = CombinedModel(combined_model_params)

## warning! the multiple combined tests will overwite each ohter's results
#'''
############## Section of regular validation #######################
train_index, valid_index = create_validation_index(train, 0.5, dep_var_name, True)  
valid_data = train.ix[valid_index]
tmp_train  = train.ix[train_index]

combined_model.fit(train, dep_var_name)
pred_df = combined_model.predict(valid_data)
print 'MCC score from validation: ', MCC(valid_data[dep_var_name], pred_df)
#print score_MCC(valid_data[dep_var_name], pred_df)
#'''


#'''
############## Section of using cross validation #######################
## cross-validate any combined model
results = cross_validate_model(train, dep_var_name, combined_model, score_MCC, 3)
print results
#'''


############## Section of cross_valiate fit #######################
combined_model.cross_vlidate_fit(train, dep_var_name)
pred_df = combined_model.predict(valid_data, score_conversion_type='A')
print 'MCC score from cross_valiate_fit: ', MCC(valid_data[dep_var_name], pred_df)




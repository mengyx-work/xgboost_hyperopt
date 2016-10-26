import pandas as pd
import numpy as np
import os, sys, time, random
from os.path import join
import yaml
import cPickle as pickle

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
from utils.bosch_functions import load_processed_bosch_data
from utils.utils_functions import print_colors
from utils.models import CombinedModel
from utils.validation_tools import score_MCC, MCC, create_validation_index, cross_validate_model
from utils.validation_tools import combine_prediction_results_for_combined_models

dep_var_name = 'Response'
project_name = 'processed_subsample_50k_data_xgb'

start_time = time.time()
data_path = '/home/ymm/kaggle/bosch_data/bosch_processed_data'
train_file_name = 'bosch_combined_train_data.csv'
#test_file_name = 'FE_raw_test_num_dat_data.csv'

## subsample training data
tot_row_num = 1183747
num_rows = 50000
skip = sorted(random.sample(xrange(1,tot_row_num + 1),tot_row_num - num_rows))
train = pd.read_csv(join(data_path, train_file_name), index_col='Id', skiprows=skip)
print 'shape of training data is {}'.format(train.shape)

## load the full training data
#train = pd.read_csv(join(data_path, train_file_name), index_col='Id')

## params for combined model
raw_models_yaml_file    = 'raw_combined_models.yml'
raw_models_yaml_path    = './'
trained_model_yaml_file = 'trained_combined_model.yml'
project_path            = './cross_validate_{}_models'.format(project_name)

## train the comined model
combined_model_params = {}
combined_model_params['raw_models_yaml_file']   = raw_models_yaml_file
combined_model_params['project_path']           = project_path
combined_model_params['models_yaml_file']       = trained_model_yaml_file
combined_model_params['raw_models_yaml_path']   = raw_models_yaml_path 

## build the combined model
combined_model = CombinedModel(combined_model_params)
combined_model.cross_vlidate_fit(train, dep_var_name, fold_num=3)
print '{} finish training and prediction on data bin: {}, using {} minutes'.format(print_colors.GREEN, project_name, round((time.time() - start_time)/60, 1))


import pandas as pd
import numpy as np
import os, sys, time
import yaml
import cPickle as pickle

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
from utils.bosch_functions import load_processed_bosch_data
from utils.utils_functions import print_colors
from utils.models import CombinedModel
from utils.validation_tools import score_MCC, MCC, create_validation_index, cross_validate_model

dep_var_name = 'Response'
tot_bins = [str(x) for x in range(2)] + ['NaN']

## 3 bins data
project_yml_path    = '/mnt/home/ymm/kaggle/compete/current'
data_path           = '/home/ymm/kaggle/bosch_data/bosch_complete_processed_3_bins_data'
data_yaml_file      = 'complete_subset_data_3_bins_dict.yml'

for bin_index in tot_bins:
    '''
    all the combined models share the same configuration.

    each combined model will be built based on
    a unique set of data, prediction will be conducted 
    on the similar set of test data
    '''

    print '{} \n start training and prediction on data bin: {}'.format(print_colors.GREEN, bin_index)
    start_time = time.time()
    train, test  = load_processed_bosch_data(data_path, data_yaml_file, data_index = bin_index, load_test=True)

    ## params for combined model
    raw_models_yaml_file    = 'raw_combined_models.yml'
    raw_models_yaml_path    = './'
    trained_model_yaml_file = 'trained_combined_model.yml'
    project_path            = './data_bin_{}_models'.format(bin_index)

    ## train the comined model
    combined_model_params = {}
    combined_model_params['raw_models_yaml_file']   = raw_models_yaml_file
    combined_model_params['project_path']           = project_path
    combined_model_params['models_yaml_file']       = trained_model_yaml_file
    combined_model_params['raw_models_yaml_path']   = raw_models_yaml_path 

    ## build the combined model
    combined_model = CombinedModel(combined_model_params)
    #combined_model.fit(train, dep_var_name)
    combined_model.cross_vlidate_fit(train, dep_var_name)
    pred_df = combined_model.predict(test, score_conversion_type = 'A')

    ## final output from combined model
    res_file_name = 'bosch_results_data_bin_{}.csv'.format(bin_index)
    pred_df.to_csv(res_file_name)
    print '{} finish training and prediction on data bin: {}, using {} seconds'.format(print_colors.GREEN, bin_index, round(time.time() - start_time, 0))


import pandas as pd
import numpy as np
import os, sys, time
import yaml
from random import shuffle
import cPickle as pickle

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
from utils.bosch_functions import load_processed_bosch_data
from utils.utils_functions import print_colors
from utils.models import CombinedModel
from utils.validation_tools import score_MCC, MCC, create_validation_index, cross_validate_model

dep_var_name = 'Response'
tot_bins = [str(x) for x in range(15)] + ['NaN']

'''
## 15 bins data
project_yml_path    = '/mnt/home/ymm/kaggle/compete/current'
data_path           = '/home/ymm/kaggle/bosch_data/bosch_complete_processed_15_bins_data'
data_yaml_file      = 'bosch_processed_data_dict.yml'
'''


#'''
## 3 bins data
project_yml_path    = '/mnt/home/ymm/kaggle/compete/current'
data_path           = '/home/ymm/kaggle/bosch_data/bosch_complete_processed_3_bins_data'
data_yaml_file      = 'complete_subset_data_3_bins_dict.yml'
#'''


for bin_index in tot_bins:

    print '{} \n start training and prediction on data bin: {}'.format(print_colors.GREEN, bin_index)
    start_time = time.time()
    train, test = load_processed_bosch_data(data_path, data_yaml_file, data_index = bin_index, load_test=True)
    #train = pd.read_csv('/home/ymm/kaggle/bosch_data/bosch_complete_processed_15_bins_data/processed_totBins_16_bin_0_train.csv', index_col='Id', nrows=5000)
    #test = pd.read_csv('/home/ymm/kaggle/bosch_data/bosch_complete_processed_15_bins_data/processed_totBins_16_bin_0_test.csv', index_col='Id', nrows=5000)

    ## big loop on the subsets of data
    labels = train[dep_var_name]
    slice_num = 10

    negative_index = train.index[labels == 0].tolist()
    positive_index = train.index[labels == 1].tolist()

    shuffle(negative_index)
    index_list = [int(1.*i/slice_num*len(negative_index)) for i in range(slice_num)]
    index_list.append(len(negative_index))

    for i in range(slice_num):
        data_index = positive_index[:]
    	tmp_negative_index = negative_index[index_list[i]:index_list[i+1]]
    	data_index.extend(tmp_negative_index)

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
        tmp_train = train.ix[data_index]
        if i == 0:
            combined_model.fit(tmp_train, dep_var_name)
        else:
            combined_model.fit(tmp_train, dep_var_name, append_models=True)

    ## outside the slice loop, predict by using the combined models
    pred_df = combined_model.predict(test, score_conversion_type='B')

    ## final output from combined model
    res_file_name = 'bosch_results_data_bin_{}.csv'.format(bin_index)
    pred_df.to_csv(res_file_name)
    print '{} finish training and prediction on data bin: {}, using {} seconds'.format(print_colors.GREEN, bin_index, round(time.time() - start_time, 0))


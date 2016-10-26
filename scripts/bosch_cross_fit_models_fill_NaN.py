import pandas as pd
import numpy as np
import os, sys, time, random
from os.path import join
import yaml
import cPickle as pickle

sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
from utils.bosch_functions import load_processed_bosch_data
from utils.utils_functions import print_colors
from utils.feature_engineering import encode_categorical_by_dep_var

from utils.models import CombinedModel
from utils.validation_tools import score_MCC, MCC, create_validation_index, cross_validate_model
from utils.validation_tools import combine_prediction_results_for_combined_models
from utils.validation_tools import get_combinedFeaImp_fromProj


dep_var_name = 'Response'
project_name = 'processed_subsample_50k_data_rf'

data_path = '/home/ymm/kaggle/bosch_data/bosch_processed_data'
train_file_name = 'bosch_combined_train_data.csv'

def get_selected_features(selected_features, feature_set, train_features):
    selected_subset_feature = [col for col in feature_set if col in selected_features]
    for col in selected_subset_feature:
        if col not in train_features:
            print 'column {} does not exist in train'.format(col)
    return selected_subset_feature


with open(os.path.join(data_path, 'bosch_feature_dict.yml'), 'r') as yml_stream:
        models_dict = yaml.load(yml_stream)

feature_data_path = '/home/ymm/kaggle/xgboost_hyperopt/scripts/xgb_model_features_0'
combined_feature_importance = get_combinedFeaImp_fromProj(feature_data_path)
selected_features = combined_feature_importance.index.tolist()

## subsample training data
tot_row_num = 1183747
num_rows = 5000
skip = sorted(random.sample(xrange(1,tot_row_num + 1),tot_row_num - num_rows))
train = pd.read_csv(join(data_path, train_file_name), index_col='Id', skiprows=skip)
print 'shape of training data is {}'.format(train.shape)

station_features = get_selected_features(selected_features, models_dict['station'], train.columns.tolist())
num_features = get_selected_features(selected_features, models_dict['num'], train.columns.tolist())
cat_features = get_selected_features(selected_features, models_dict['cat'], train.columns.tolist())
dat_features = get_selected_features(selected_features, models_dict['date'], train.columns.tolist())

train_stat = train[station_features]
train_cat = train[cat_features]
train_num = train[num_features]
train_dat = train[dat_features]
print train_stat.shape, train_cat.shape, train_num.shape, train_dat.shape

num_missing_value= -1.5
dat_missing_value = -1.
station_fillna_value = 9999999

train_stat = train_stat.fillna(station_fillna_value)
train_num = train_num.fillna(num_missing_value)
train_dat = train_dat.fillna(dat_missing_value)


start_time = time.time()
train_cat['Response'] = train['Response']
encode_columns_dict = encode_categorical_by_dep_var(train_cat, dep_var_column='Response', fill_missing=True, fill_missing_value = 9999999)
train_cat.drop('Response', axis=1, inplace=True)
print 'finish encoding categorical features using {} seconds'.format(round(time.time() - start_time, 0))

print train_dat.isnull().sum().sum(), train_num.isnull().sum().sum(), train_cat.isnull().sum().sum(), train_stat.isnull().sum().sum()
combined_train = pd.concat([train_dat, train_num, train_cat, train_stat], axis=1)
combined_train[dep_var_name] = train[dep_var_name]
print combined_train.shape, train.shape

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
combined_model.cross_vlidate_fit(combined_train, dep_var_name, fold_num=2)
print '{} finish training and prediction on data bin: {}, using {} minutes'.format(print_colors.GREEN, project_name, round((time.time() - start_time)/60, 1))


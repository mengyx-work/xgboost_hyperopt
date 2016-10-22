import os, sys, time
import pandas as pd
import numpy as np
from os.path import join, isfile


def combine_feature_importance_files(data_path, fea_name='feature', thres_name = None, thres = 10):
    csv_files = [f for f in os.listdir(data_path) if '.csv' in f]
    fea_imp = None
    file_counter = -1
    score_columns = []
    norm_score_columns = []
    
    for file_name in csv_files:
        data = pd.read_csv(join(data_path, file_name), index_col=0)
        
        if thres_name is not None:
            data = data.loc[data[thres_name] > thres]
            
        data = data.set_index(fea_name)
        print data.shape
        file_counter += 1
        data.columns = ['{}_{}'.format(column, file_counter) for column in data.columns]
        score_columns.append('{}_{}'.format('fscore', file_counter))
        norm_score_columns.append('{}_{}'.format('norm_fscore', file_counter))
        if fea_imp is None:
            fea_imp = data
        else:
            fea_imp = pd.merge(fea_imp, data, how='outer', left_index=True, right_index=True)
    
    fea_imp['fscore_sum'] = fea_imp[score_columns].sum(axis=1)
    fea_imp['norm_fscore_sum'] = fea_imp[norm_score_columns].sum(axis=1)
    return fea_imp


def combine_data_byColumns(data_path, csv_files, selected_features, idx_col_name, dep_var_name):
    '''
    assume the csv_files includes only the training
    csv files and corresponding test csv files share
    the same file name pattern
    '''
    train, test = None, None
    for train_csv in csv_files:
        start_time = time.time()
        test_csv = train_csv.replace('train', 'test')
        if not isfile(join(data_path, train_csv)):
            raise ValueError('fail to locate the training data {}'.format(join(data_path, train_csv)))
        if not isfile(join(data_path, test_csv)):
            raise ValueError('fail to locate the test data {}'.format(join(data_path, test_csv)))

        train_columns = pd.read_csv(join(data_path, train_csv), index_col=False, nrows=1)
        selected_columns = [col for col in train_columns if col in selected_features]
        tmp_train = pd.read_csv(join(data_path, train_csv), index_col=idx_col_name, usecols=selected_columns)

        if dep_var_name in selected_columns:
            selected_columns.remove(dep_var_name)
        tmp_test = pd.read_csv(join(data_path, test_csv), index_col=idx_col_name, usecols=selected_columns)

        if  train is None:
            train = tmp_train
        else:
            train = pd.concat([train, tmp_train], axis=1)
        if test is None:
            test = tmp_test
        else:
            test = pd.concat([test, tmp_test], axis=1)
        print 'finish loading the data {}, {} with shape {}, {}'.format(train_csv, test_csv, tmp_train.shape, tmp_test.shape)
        print 'train, test shape: {}, {}, using {} minutes'.format(train.shape, test.shape, round((time.time()-start_time)/60, 2))

    return train, test



feature_data_path = '/home/ymm/kaggle/xgboost_hyperopt/scripts/xgb_model_features_0'
dep_var_name = 'Response'
idx_col_name = 'Id'


data_path = '/home/ymm/kaggle/bosch_data/bosch_processed_data'
csv_files = ['bosch_train_categorical_features.csv', 'bosch_train_date_features.csv',
             'bosch_train_numerical_features.csv', 'bosch_train_station_features.csv']

combined_feature_importance = combine_feature_importance_files(feature_data_path)
selected_features = combined_feature_importance.index.tolist()
selected_features.append(dep_var_name)
selected_features.append(idx_col_name)

train, test = combine_data_byColumns(data_path, csv_files, selected_features, idx_col_name, dep_var_name) 
start_time = time.time()
train.to_csv('bosch_combined_train_data.csv')
test.to_csv('bosch_combined_test_data.csv')
print 'finish writing data to csv using {} minutes'.format(round((time.time()-start_time)/60, 2))

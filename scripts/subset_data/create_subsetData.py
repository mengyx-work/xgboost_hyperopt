import os, sys
import pandas as pd
import numpy as np
from os.path import join, isfile

fea_data_path = '/home/ymm/kaggle/bosch/full_data_FE_cross_fit_xgb_models/final_xgb_models/combined_xgb_models_038'
data_path = '/home/ymm/kaggle/bosch_data/bosch_processed_data'
train_file_name = 'bosch_combined_train_data.csv'
index_col_name = 'Id'
dep_var_name = 'Response'

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


combined_imp = combine_feature_importance_files(fea_data_path)
print 'the shape of combined feature set is {}'.format(combined_imp.shape)
selected_features = combined_imp[combined_imp.isnull().sum(axis=1) == 0].index.tolist()
selected_features.append(index_col_name)
selected_features.append(dep_var_name)

train = pd.read_csv(join(data_path, train_file_name), index_col=index_col_name, usecols=selected_features)
train.to_csv('./selected_combined_train.csv')


import pandas as pd
import numpy as np
import time, os, sys
from . import data_munge

def load_data_by_index(skipped_train_row_num, skipped_test_row_num, train_data_file, test_data_file, data_path):

    data_path = '/home/ymm/bosch/'

    train_num_file   = 'train_numeric.csv'
    train_cat_file   = 'train_categorical.csv'
    train_date_file  = 'train_date.csv'
    test_num_file    = 'test_numeric.csv'
    test_cat_file    = 'test_categorical.csv'
    test_date_file   = 'test_date.csv'

    start_time_column_name = 'L0_S0_D1'
    id_column_name = 'Id'
    dep_var_name = 'Response'

    '''
    function to load a subset of Bosch data based on the skipped_row_num lists.
    three separate csv files are loaded: numerical, categorical and date for 
    train and test data
    '''

    start_time = time.time()
    train_date = pd.read_csv(data_path + train_date_file, index_col='Id', skiprows=skipped_train_row_num)
    train_num = pd.read_csv(data_path + train_num_file, index_col='Id', skiprows=skipped_train_row_num)
    train_cat = pd.read_csv(data_path + train_cat_file, index_col='Id', skiprows=skipped_train_row_num)
    test_date = pd.read_csv(data_path + test_date_file, index_col='Id', skiprows=skipped_test_row_num)
    test_num = pd.read_csv(data_path + test_num_file, index_col='Id', skiprows=skipped_test_row_num)
    test_cat = pd.read_csv(data_path + test_cat_file, index_col='Id', skiprows=skipped_test_row_num)
    end_time = time.time()
    print 'data loading takes ', round((end_time - start_time), 2), 'seconds'

    
    ## process the date data
    process_date_data(train_date, test_date, start_time_column_name)
    print 'finish processing date data ...'
    
    ## process the numerical data
    data_munge.remove_single_value_columns(train_num, test_num)
    print 'finish processing numerical data ...'

    ## process categorical data
    data_munge.remove_single_value_columns(train_cat, test_cat)
    encodeed_train_cat, encoded_test_cat = data_munge.encode_columns(train_cat, test_cat, True)
    print 'finish processing categorical data ...'

    ## combine the data and save into csv files
    combined_train = pd.concat([encodeed_train_cat, train_num, train_date], axis=1)
    combined_test = pd.concat([encoded_test_cat, test_num, test_date], axis=1)
    
    combined_train.to_csv(train_data_file)
    combined_test.to_csv(test_data_file)


def encode_categorical_data(train, test, fill_missing = False):
    '''
    '''
    le = LabelEncoder()
    
    if fill_missing:
        train = train.fillna(value='missing')
        test = test.fillna(value='missing')
    
    counter = 0
    start_time = time.time()
    for col, dtype in zip(train.columns, train.dtypes):
        if dtype == 'object':
            le.fit(pd.concat([train[col], test[col]], axis=0))
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
                              
        counter += 1
        if counter % 20 == 0:
            print '{} out of {} is process...'.format(str(counter), str(train.shape[1]))
                              
    end_time = time.time()
    print 'encoding process takes ', round((end_time - start_time)), 'seconds'
    
    return train, test
    

def process_date_data(train_date, test_date, start_time_column_name):
    print 'raw date data dimension: ', train_date.shape, test_date.shape
    train_date['start_time'] = train_date[start_time_column_name]
    test_date['start_time'] = test_date[start_time_column_name]
    single_value_column_names = []

    for column in train_date.columns:
        if column != 'start_time':
            train_date[column] = train_date[column] - train_date['start_time']
            test_date[column] = test_date[column] - test_date['start_time']
        if len(train_date[column].unique()) == 1:
            single_value_column_names.append(column)

    ## drop single-valued columns
    train_date.drop(single_value_column_names, axis=1, inplace=True)
    test_date.drop(single_value_column_names, axis=1, inplace=True)
    print 'processed date data dimension: ', train_date.shape, test_date.shape


def process_train_date_date(train_date):
    train_date['start_time'] = train_date[start_time_column_name]
    single_value_column_names = []

    for column in train_date.columns:
        if column != 'start_time':
            train_date[column] = train_date[column] - train_date['start_time']
        if len(train_date[column].unique()) == 1:
            single_value_column_names.append(column)
    
    print 'before remvoing single_value column:', train_date.shape
    train_date.drop(single_value_column_names, axis=1, inplace=True)
    print 'after remvoing single_value column:', train_date.shape
   


## assuming the model output is the probability of being default,
## then this probability can be used for ranking. Then using the fraction of
## default in validation data to assign the proper threshold to the prediction
def score_MCC(ground_truth, scores):
    tmp_ground_truth = ground_truth
    fault_frac = tmp_ground_truth.mean()
    print 'score shape:', scores.shape, 
    print 'mean of groud truth:', fault_frac
    thres_value = np.percentile(scores, 100.*(1-fault_frac), axis=0)
    print 'threshold value:', thres_value
    binary_scores = scores > thres_value
    binary_scores = binary_scores.astype(int)
    ## convert to sk-learn format
    np.place(binary_scores, binary_scores==0, -1)
    np.place(tmp_ground_truth, tmp_ground_truth==0, -1)
    #print ground_truth
    #print binary_scores
    return matthews_corrcoef(tmp_ground_truth, binary_scores)
 

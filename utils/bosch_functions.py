import pandas as pd
import numpy as np
import time, os, sys

def load_data_by_index(skipped_train_row_num, skipped_test_row_num, train_data_file, test_data_file):

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
    
    ## process categorical data
    remove_single_value_categorical_columns(train_cat, test_cat)
    encode_categorical_data(train_cat, test_cat, True)
    print 'finish processing categorical data ...'

    ## combine the data and save into csv files
    combined_train = pd.concat([train_cat, train_num, train_date], axis=1)
    combined_test = pd.concat([test_cat, test_num, test_date], axis=1)
    
    combined_train.to_csv(train_data_file)
    combined_test.to_csv(test_data_file)


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
 

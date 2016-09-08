import pandas as pd
import numpy as np
import time, os, sys
from scipy.stats.mstats import mquantiles
from . import data_munge

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


def create_grouped_index_df(bin_num):
    ## load the labels and start_time column for train and test data
    start_time = time.time()
    train_labels = pd.read_csv(data_path + train_num_file, index_col='Id', usecols=['Id', dep_var_name])
    train_date_start_columm = pd.read_csv(data_path + train_date_file, index_col='Id', usecols=['Id', start_time_column_name])
    test_date_start_columm = pd.read_csv(data_path + test_date_file, index_col='Id', usecols=['Id', start_time_column_name])
    end_time = time.time()
    print 'data loading takes ', round((end_time - start_time), 1), ' seconds.'

    ## join the start_time with labels, then drop the NaN in start_time
    labeled_start_time = pd.merge(train_labels, train_date_start_columm, how='left', left_index=True, right_index=True)
    ## this labeled_start_time dataFrame doesn't contain the NaN, therefore it can be directly used for calculating the mquantiles
    labeled_start_time = labeled_start_time[~labeled_start_time[start_time_column_name].isnull()]


    ##section to subset the data by start_time
    prob_list = [1.*i/bin_num for i in range(1, bin_num)]
    quantile_values = mquantiles(labeled_start_time[start_time_column_name], prob=prob_list)

    bins = [labeled_start_time[start_time_column_name].min()]
    bins.extend(quantile_values)
    bins.append(labeled_start_time[start_time_column_name].max())
    bin_names = [str(i) for i in range(len(bins)-1)]

    ## cut the entire dataframe into different time_windows by start_time
    tmp_train = train_date_start_columm.copy()
    tmp_test = test_date_start_columm.copy()

    tmp_train['time_window_num'] = pd.cut(tmp_train[start_time_column_name], bins, labels=bin_names)
    tmp_test['time_window_num'] = pd.cut(tmp_test[start_time_column_name], bins, labels=bin_names)
    ## create a row number column, start index is 1
    tmp_train['row_num'] = range(1, (tmp_train.shape[0] + 1))
    tmp_test['row_num'] = range(1, (tmp_test.shape[0] + 1))

    return tmp_train, tmp_test, bins, bin_names 



def subset_complete_data_by_index(skipped_train_row_num, skipped_test_row_num, train_data_file, test_data_file):

    '''
    function that load the numerical, date and categorical types
    data for train and test by skipped_row_num.
    
    process them in different ways:
    1. numerical: remove single-valued column and fill up NaN
    2. categorical: remove single-valued column, fill NaN with 'missing' and encode them
    3. date: normalize the date/time to the start_time and remove single-valued columns

    save data into csv files
    '''

    train_date = pd.read_csv(data_path + train_date_file, index_col='Id', skiprows=skipped_train_row_num)
    train_num  = pd.read_csv(data_path + train_num_file,  index_col='Id', skiprows=skipped_train_row_num)
    train_cat  = pd.read_csv(data_path + train_cat_file,  index_col='Id', skiprows=skipped_train_row_num)
    test_date  = pd.read_csv(data_path + test_date_file,  index_col='Id', skiprows=skipped_test_row_num)
    test_num   = pd.read_csv(data_path + test_num_file,   index_col='Id', skiprows=skipped_test_row_num)
    test_cat   = pd.read_csv(data_path + test_cat_file,   index_col='Id', skiprows=skipped_test_row_num)
    
    ## process the date data
    process_date_data(train_date, test_date, start_time_column_name)

    if not all(train_date.dtypes == 'float64'):
        print 'date train data has non-numerical columns!'
        sys.exit(0)
        
    if not all(test_date.dtypes == 'float64'):
        print 'date test data has non-numerical columns!'
        sys.exit(0)

    train_date, test_date = data_munge.replace_missing_with_fix_value(train_date, test_date, -1.) 
    print 'finish processing date data ...'
    
    ## process the numerical data
    data_munge.remove_single_value_columns(train_num, test_num)
    replace_missing_numerical_value(train_num, test_num)
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


def replace_missing_numerical_value(train, test):
    for col, dtype in zip(train.columns, train.dtypes):
        if dtype == 'float64' or dtype == 'int64':
            ## to avoid the dep_var/label column
            if col != dep_var_name:
                combined_series = pd.concat([train[col], test[col]], axis=0)
            else:
                continue

            ## select the fill_value for NaN
            if sum(combined_series.isnull()) == 0:
                continue
            if combined_series.max() < 1.:
                fill_value = 1.
            elif combined_series.min() > -1.:
                fill_value = -1.
            elif 0. not in combined_series.values:
                fill_value = 0.
            else:
                print 'failed to find proper value to fill missing value in column: ', col
                sys.exit(0)

            train[col] = train[col].fillna(value = fill_value)
            test[col] = test[col].fillna(value = fill_value)

        else:
            print 'none numerical column is found: ', col
            sys.exit(0)


'''
def encode_categorical_data(train, test, fill_missing = False):

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
'''
    

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
 

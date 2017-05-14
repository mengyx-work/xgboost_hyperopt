import os, sys, time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold



def remove_single_value_columns(train, targe_column=None, test=None):

    print 'raw train data dimension: ', train.shape

    if test is not None:
        print 'raw test data dimension: ', test.shape
        if len(test.columns) != len(train.columns) and targe_column is None:
            print 'test has #columns {} and train has #columns {}.'.format(test.shape[1], train.shape[1])
        test_columns = test.columns.tolist()
        train_columns = train.columns.tolist()
        if targe_column is not None:
            train_columns.remove(targe_column)
        if train_columns != test_columns:
            raise ValueError('train and test have different columns!')

    single_value_column_names = []
    for col in train.columns:
        if len(train[col].unique()) == 1:
            single_value_column_names.append(col)

    train.drop(single_value_column_names, axis=1, inplace=True)
    print 'processed train data dimension: ', train.shape

    if test is not None:
        for col in single_value_column_names:
            if col not in test.columns:
                print 'warning! column {} does not exist in test data.'.format(col)
        test.drop(single_value_column_names, axis=1, inplace=True)
        print 'processed test data dimension: ', test.shape



def check_dataFrame_numerical_columns(train, test):
    
    if not train.isnull().sum() == 0 or not test.isnull().sum():
        return False

    for col, dtype in zip(train.columns, train.dtypes):
        if dtype != 'float64' or dtype != 'int64':
            print 'column ', col, ' is not numerical....'
            return False

    return True



def encode_columns(train, test=None, fill_missing = False):
    '''
    encoding is an extemely slow process
    So only use the training data to trian the encoder
    '''
    le = LabelEncoder()

    ## this step creates separate train and test dataFrame
    if fill_missing:
        train = train.fillna(value='missing')
        if test is not None:
            test = test.fillna(value='missing')

    counter = 0
    start_time = time.time()
    for col in train.columns:
        if test is not None:
            le.fit(pd.concat([train[col], test[col]], axis=0))
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
        else:
            le.fit(train[col])
            train[col] = le.transform(train[col])

        counter += 1
        if counter % 20 == 0:
            print '{} out of {} is processed...'.format(str(counter), str(train.shape[1]))

    end_time = time.time()
    print 'encoding process takes ', round((end_time - start_time)), 'seconds'

    ## train and test are newly created
    if test is not None:
        return train, test
    else:
        return train

 

def encode_categorical_data(train, test, fill_missing = False):
    '''
    encoding is an extemely slow process
    So only use the training data to trian the encoder
    '''
    le = LabelEncoder()

    ## this step creates separate train and test dataFrame
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
            print '{} out of {} is processed using {} seconds...'.format(str(counter), str(train.shape[1]), round((time.time() - start_time), 0))

    end_time = time.time()
    print 'encoding process takes ', round((end_time - start_time)), 'seconds'

    ## train and test are newly created
    return train, test

      
def encode_categorical_columns_single_df(df, fill_missing = False):
    le = LabelEncoder()
    if fill_missing:
        df = df.fillna(value='missing')
        
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == 'object':
            df[col] = le.fit_transform(df[col])

    return df


def replace_missing_with_fix_value(train, test=None, missing_value=0.):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for col, dtype in zip(train.columns, train.dtypes):
        if dtype == 'float64' or dtype == 'int64':
            train_df[col] = train[col].fillna(value = missing_value)
            if test is not None:
                test_df[col] = test[col].fillna(value = missing_value)

    if test is not None:
        return train_df, test_df
    else:
        return train_df
          
                
def replace_missing_with_random_sample(df):
    for col in df.columns:
        boolean_nan_index = df[col].isnull()
        if sum(boolean_nan_index) > 0:
            none_nan_df = df[col][~boolean_nan_index]
            df[col][boolean_nan_index] = none_nan_df.sample(n=sum(boolean_nan_index), replace=True).tolist()

        

def remove_highly_missing_col(df, mis_frac_thres = 0.95):
    highly_missing_col_names = []
    for col in df.columns:
        if 1.*sum(df[col].isnull()) / df.shape[0] >= mis_frac_thres:
            highly_missing_col_names.append(col)
    
    df.drop(highly_missing_col_names, axis=1, inplace=True)


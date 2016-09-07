import os, sys, time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def remove_single_value_categorical_columns(train, test):
    print 'raw categorical data dimension: ', train.shape, test.shape
    single_value_column_names = []
    for col in train.columns:
        if len(train[col].unique()) == 1:
            single_value_column_names.append(col)

    train.drop(single_value_column_names, axis=1, inplace=True)
    test.drop(single_value_column_names, axis=1, inplace=True)
    print 'processed categorical data dimension: ', train.shape, test.shape



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

    ## idealy combine the train and test
    #combined = pd.concat([train, test], axis=0)
    counter = 0
    start_time = time.time()
    for col, dtype in zip(train.columns, train.dtypes):
        if dtype == 'object':
            le.fit(pd.concat([train[col], test[col]], axis=0))
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])

        counter += 1
        if counter % 20 == 0:
            print '{} out of {} is processed...'.format(str(counter), str(train.shape[1]))

    end_time = time.time()
    print 'encoding process takes ', round((end_time - start_time)), 'seconds'

    ## train and test are newly created
    return train, test


## inplace to remove single value columns
def remove_single_value_columns(df):
    single_value_column_names = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            single_value_column_names.append(col)
    print 'before remvoing single_value column:', df.shape
    df.drop(single_value_column_names, axis=1, inplace=True)
    print 'after remvoing single_value column:', df.shape

      

def encode_categorical_columns_single_df(df, fill_missing = False):
    le = LabelEncoder()
    if fill_missing:
        df = df.fillna(value='missing')
        
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == 'object':
            df[col] = le.fit_transform(df[col])
            


def replace_missing_with_fix_value(df, missing_value):
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == 'float64':
            df[col] = df[col].fillna(value = missing_value)
          
                
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


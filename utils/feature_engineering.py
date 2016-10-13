from sklearn import preprocessing
import pandas as pd
import numpy as np

def BasicDate_FeatureEngineering(tmp_train_dat, start_time_column=None):
    ## feature engineering on the date features
    encoder = preprocessing.LabelEncoder()
    column_names = tmp_train_dat.columns.tolist()
    column_names.append('NaN')
    encoder.fit(column_names)
    dat_new_fea = pd.DataFrame()
    
    if 'L0_S0_D1' in tmp_train_dat.columns:
        dat_new_fea['start_time'] = tmp_train_dat['L0_S0_D1']
        
    dat_new_fea['time_sum'] = tmp_train_dat.sum(axis=1)
    dat_new_fea['time_mean'] = tmp_train_dat.mean(axis=1)
    dat_new_fea['dat_nan_count'] = tmp_train_dat.isnull().sum(axis=1)
    dat_new_fea['max_time'] = tmp_train_dat.max(axis=1)
    dat_new_fea['min_time'] = tmp_train_dat.min(axis=1)
    dat_new_fea['dat_max_min_diff'] = dat_new_fea['max_time'] - dat_new_fea['min_time']
    dat_new_fea['dat_max_min_ratio'] = dat_new_fea['min_time'] / dat_new_fea['max_time']

    dat_new_fea['dat_idxmax'] = tmp_train_dat.idxmax(axis=1)
    dat_new_fea['dat_idxmax'].fillna('NaN', inplace=True)
    dat_new_fea['dat_idxmax'] = encoder.transform(dat_new_fea['dat_idxmax'])
    dat_new_fea['dat_idxmin'] = tmp_train_dat.idxmin(axis=1)
    dat_new_fea['dat_idxmin'].fillna('NaN', inplace=True)
    dat_new_fea['dat_idxmin'] = encoder.transform(dat_new_fea['dat_idxmin'])

    return dat_new_fea



def NumericalFeatureEngineering(df, col_ignore = ['Response']):
    '''
    function to create general engineering features
    for numerical columns
    '''
    tmp_df = df.loc[:, ~df.columns.isin(col_ignore)]
    new_fea_df = pd.DataFrame()
    encoder = preprocessing.LabelEncoder()
    column_names = tmp_df.columns.tolist()
    column_names.append('NaN')
    encoder.fit(column_names)
    
    new_fea_df['num_mean'] = tmp_df.mean(axis=1)
    new_fea_df['num_sum'] = tmp_df.sum(axis=1)
    new_fea_df['num_max'] = tmp_df.max(axis=1)
    new_fea_df['num_min'] = tmp_df.min(axis=1)
    new_fea_df['num_max_min_ratio'] = new_fea_df['num_min'] / new_fea_df['num_max']
    new_fea_df['num_max_min_ratio'] = new_fea_df['num_max_min_ratio'].replace([np.inf, -np.inf], np.NaN)
    new_fea_df['num_nan_col_count'] = tmp_df.isnull().sum(axis=1)
    new_fea_df['num_reg_col_count'] = tmp_df.shape[1] - tmp_df.isnull().sum(axis=1)
    new_fea_df['num_idxmax'] = tmp_df.idxmax(axis=1)
    new_fea_df['num_idxmax'].fillna('NaN', inplace=True)
    new_fea_df['num_idxmax'] = encoder.transform(new_fea_df['num_idxmax'])
    new_fea_df['num_idxmin'] = tmp_df.idxmin(axis=1)
    new_fea_df['num_idxmin'].fillna('NaN', inplace=True)
    new_fea_df['num_idxmin'] = encoder.transform(new_fea_df['num_idxmin'])
    return new_fea_df



def getRelativeTimeColumns(series):
    '''
    normalize the time features by
    the start_time, the first none-NaN
    value
    '''
    if series[0] == np.NaN:
        start_time = series.dropna().index[0]
    else:
        start_time = series[0]
    new_series = series - start_time
    return new_series
   


def getTimeSteps(series, unique_value_counts=6):
    '''
    in each row/series, use the sorted value_count
    to find the time steps and use the value, counts
    and column_index as features
    '''
    value_counts = series.value_counts()
    value_counts.sort_index(inplace=True)

    if 0. in value_counts.index:
        value_counts = value_counts[value_counts.index != 0.]
        
    available_counts = value_counts.shape[0]
    feature_array = []
    for i in xrange(unique_value_counts):
        if i < available_counts:
            date_value = value_counts.index[i]
            counts = value_counts[date_value]
            first_index = series[series == date_value].index[0]
            avg_time_cost = date_value / counts
            feature = [date_value, counts, avg_time_cost, first_index]
        else:
            feature = [np.NaN, 0, 0, 'NaN']
        feature_array.extend(feature)

    return pd.Series(feature_array)



def getTimeChangeColumns(series):
    '''
    function to find the first column that has
    a different date value compared to the start_time.

    'start_time' is defined to be the value of first
    column if it's not NaN.

    If 'sart_time' is NaN, the first none-NaN value 
    is treated as the different value

    'first_num_value' is the first none-NaN numerical value
    '''
    start_time = series[0]
    tmp_series = series.dropna()
    if start_time == np.NaN:
        first_index     = tmp_series.index[0]
        last_index      = tmp_series.index[-1]
        first_id_value  = tmp_series[first_index]
        last_id_value   = tmp_series[last_index]
        first_num_value = first_id_value
        time_diff       = last_id_value - first_id_value
        time_ratio      = last_id_value / first_id_value
        return pd.Series([first_index, last_index, time_diff, time_ratio, 
                          first_id_value, last_id_value, first_num_value])
    else:
        first_num_value = start_time
        if np.sum(tmp_series != start_time) == 0:
            return pd.Series(['NaN', 'NaN', np.NaN, np.NaN, np.NaN, np.NaN, first_num_value])
        else:
            first_index     = tmp_series.index[tmp_series != start_time][0]
            last_index      = tmp_series.index[tmp_series != start_time][-1]
            first_id_value  = tmp_series[first_index]
            last_id_value   = tmp_series[last_index]
            time_diff       = last_id_value - first_id_value
            time_ratio      = last_id_value / first_id_value

            return pd.Series([first_index, last_index, time_diff, time_ratio,
                              first_id_value, last_id_value, first_num_value])



def build_sortedData_indexDiff(train_test, dat_new_fea, column_list):
    for column in column_list:
        train_test = train_test.sort_values(by=[column, 'index'], ascending=True)
        dat_new_fea['{}_index_diff_0'.format(column)] = train_test['index'].diff().fillna(9999999).astype(int)
        dat_new_fea['{}_index_diff_1'.format(column)] = train_test['index'].iloc[::-1].diff().fillna(9999999).astype(int)



def build_IndexFeatures(train, test=None, start_time_column = 'start_time'):
    '''
    function uses a combined DataFrame of train and test to build
    index/ordder based on different columns.
    '''
    expected_columns = ['first_time_value', 'last_time_value', 'time_ratio_value',
                        'first_date_value', 'start_time']

    if test is not None:
        train_test = pd.concat([train[expected_columns], test[expected_columns]], axis=0)
    else:
        train_test = train[expected_columns]

    dat_new_fea = pd.DataFrame()
    train_test['index']              = train_test.index
    dat_new_fea['index']             = train_test['index']
    dat_new_fea['first_time_index']  = train_test['first_time_value'].argsort() + 1
    dat_new_fea['last_time_index']   = train_test['last_time_value'].argsort() + 1
    dat_new_fea['index_ratio']       = dat_new_fea['first_time_index'] / dat_new_fea['last_time_index']

    if start_time_column in train_test.columns:
        dat_new_fea['start_time_diff']          = train_test['start_time'].diff()
        dat_new_fea['start_time_index']         = train_test['start_time'].argsort() + 1
        dat_new_fea['start_time_index_ratio_1'] = dat_new_fea['first_time_index'] / dat_new_fea['index']
        dat_new_fea['start_time_index_ratio_2'] = dat_new_fea['last_time_index'] / dat_new_fea['index']

        ## Bosch approach to generate features
        build_sortedData_indexDiff(train_test, dat_new_fea, ['start_time'])
    
    dat_new_fea['time_ratio_value_index']    = train_test['time_ratio_value'].argsort() + 1
    dat_new_fea['first_time_value_index']    = train_test['first_time_value'].argsort() + 1
    dat_new_fea['first_date_value_index']    = train_test['first_date_value'].argsort() + 1
    dat_new_fea['first_date_value_index_ratio_1'] = dat_new_fea['first_time_index'] / dat_new_fea['index']
    dat_new_fea['first_date_value_index_ratio_2'] = dat_new_fea['last_time_index'] / dat_new_fea['index']

    '''
    learned from Bosch that sort the data by different interesting columns, 
    the relatively difference between adjacent rows can be useful
    '''
    build_sortedData_indexDiff(train_test, dat_new_fea, ['time_ratio_value', 'first_time_value', 'last_time_value', 'first_date_value'])
    dat_new_fea.drop('index', axis=1, inplace=True)
  
    return dat_new_fea




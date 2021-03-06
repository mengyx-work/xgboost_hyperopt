import pandas as pd
import numpy as np
import os, sys, time, random
from os.path import isfile, join
from random import shuffle
from sklearn.metrics import matthews_corrcoef
from sklearn.cross_validation import StratifiedKFold
import itertools




def get_combinedFeaImp_fromProj(data_path, only_common_fea=False, fea_name='feature', thres_name=None, thres=10):
    '''function to collect feature importances
    from multiple models as a combined model project/folder

    options are provided to selected features
    1. based on threshold on certain column
    2. the common feature among all models
    '''
    csv_files = [f for f in os.listdir(data_path) if '.csv' in f]
    fea_imp = None
    file_counter = -1
    score_columns = []
    norm_score_columns = []
    
    for file_name in csv_files:
        data = pd.read_csv(join(data_path, file_name), index_col=0)
        
           
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

    ## select only features that are used in all models
    if only_common_fea:
        common_fea_boolean_index = fea_imp.isnull().sum(axis=1) == 0
        fea_imp = fea_imp[common_fea_boolean_index]

    fea_imp['fscore_sum'] = fea_imp[score_columns].sum(axis=1)
    fea_imp['norm_fscore_sum'] = fea_imp[norm_score_columns].sum(axis=1)

    ## select features based on a threshold
    if thres_name is not None:
        fea_imp = fea_imp.loc[fea_imp[thres_name] > thres]
 
    return fea_imp



def split_trainData_byTime(data, time_column_name, to_subsample=False, nrows=50000, valid_frac=0.2):
    ## use a separate data to create new train/test
    tmp_data = data.copy()
    ## to subsample the data
    if to_subsample:
        seletected_index = random.sample(tmp_data.index.tolist(), nrows)
        tmp_data = tmp_data.ix[seletected_index]

    tmp_data.loc[:, time_column_name]= pd.to_datetime(tmp_data.loc[:, time_column_name], format='%d%b%Y')
    tmp_data = tmp_data.set_index(time_column_name).sort_index()
    split_index = int((1.- valid_frac) * tmp_data.shape[0])
    return tmp_data.iloc[:split_index, :], tmp_data.iloc[split_index:, :]



def grid_search_by_fixedData(train, test, dep_var_name, model_class, eval_func, param_dict, result_file='grid_search_byFixedData_results.csv', is_xgb_model=False):
    '''
    '''
    ## flatten the parameter grid
    params_list = list(itertools.product(*param_dict.values()))
    columns_names = param_dict.keys() + ['score']
    df = pd.DataFrame(columns=columns_names)
    df.to_csv(result_file)
    row_counter = 0

    start_time = time.time()
    grid_point_counter = -1

    ## loop through the grid points  
    for param in params_list:
        grid_point_counter += 1
        
        model_params = {}
        for value, key in zip(param, param_dict.keys()):
            model_params[key] = value
            
        ## initiate new model from model_class
        if not is_xgb_model:
            model = model_class(model_params)
        else:
            model = model_class(label_name = dep_var_name, params = model_params, model_file='grid_search_xgb_model_{}'.format(grid_point_counter))

        model.fit(train, dep_var_name)
        preds = model.predict(test)
        result = eval_func(test[dep_var_name].values, preds)

        ## fill up the results DataFrame
        row_content = []
        for columns_name in columns_names[:-1]:
            row_content.append(model_params[columns_name])
        row_content.append(result)

        ## save content into csv file
        df = pd.read_csv(result_file, index_col=0)
        df.loc[row_counter] = row_content
        df.to_csv(result_file)
        row_counter += 1

        if row_counter % 10 == 0:
            print '{} grid points are finished using {} minutes'.format(row_counter, round((time.time() - start_time)/60, 0))
        



def grid_search_cross_validate_model(train, dep_var_name, model_class, eval_func, param_dict, fold_num=3, result_file='grid_search_results.csv', is_xgb_model=False):
    '''
    function to conduct grid search on models using metrics give by eval_func
    Since at each grid point, a different model is initialized; a model_class
    is provided to repeatedly initiate new model 
    '''
    ## flatten the parameter grid
    params_list = list(itertools.product(*param_dict.values()))
    columns_names = param_dict.keys() + ['avg_score', 'score_std']
    df = pd.DataFrame(columns=columns_names)
    df.to_csv(result_file)
    row_counter = 0

    start_time = time.time()
    grid_point_counter = -1

    ## loop through the grid points  
    for param in params_list:
        grid_point_counter += 1
        
        model_params = {}
        for value, key in zip(param, param_dict.keys()):
            model_params[key] = value
            
        ## initiate new model from model_class
        if not is_xgb_model:
            model = model_class(model_params)
            results = cross_validate_model(train, dep_var_name, model, eval_func, fold_num)
        else:
            xgb_model = model_class(label_name = dep_var_name, params = model_params, model_file='grid_search_xgb_model_{}'.format(grid_point_counter))
            results = xgb_model.cross_validate_fit(eval_func, train, n_folds=fold_num)

        row_content = []
        for columns_name in columns_names[:-2]:
            row_content.append(model_params[columns_name])

        row_content.append(np.mean(results))
        row_content.append(np.std(results))

        ## save content into csv file
        df = pd.read_csv(result_file, index_col=0)
        df.loc[row_counter] = row_content
        df.to_csv(result_file)
        row_counter += 1

        if row_counter % 10 == 0:
            print '{} grid points are finished using {} minutes'.format(row_counter, round((time.time() - start_time)/60, 1))
        

def combine_tuning_params(const_param_dict, tuning_param_dict):
    combined_params = const_param_dict.copy()
    for key, value in tuning_param_dict.items():
        combined_params[key] = value
    return combined_params


## helper function to put dictionary single value into a list
def list_const_params(params):
    listed_params = {}
    for key, value in params.items():
        listed_params[key] = [value]
    return listed_params



def combine_prediction_results_for_combined_models(data_path, submission_sample_path, submission_sample_file, index_col_name, res_col_name):
  '''
  function to combine a list of .csv files
  as results from a combined model in single
  folder into one combined results.
  Assuming a binary results from the combined
  model.
  '''
  csv_files = [f for f in os.listdir(data_path) if isfile(join(data_path, f)) and 'results' in f]
  print 'the collected result csv files:', csv_files
  submission_sample = pd.read_csv(join(submission_sample_path, submission_sample_file), index_col=index_col_name)
  results = pd.DataFrame()
  for csv_file in csv_files:
    ## expect the results contain index column and without header
    res = pd.read_csv(join(data_path, csv_file), index_col=0, header=None)
    results = pd.concat([results, res])

  results.index.name = index_col_name
  results.columns = [res_col_name]
  ## binary result, convert double into int
  results = results.astype('int')
  if results.shape[0] != submission_sample.shape[0]:
    sys.exit('the submission data dimension does not match sample, abort...')

  sorted_results = results.ix[submission_sample.index]
  combined_results_file = 'bosch_combined_results.csv'
  sorted_results.to_csv(join(data_path, combined_results_file))



def cross_validate_model(train_df, dep_var_name, classifier, eval_func, fold_num=2):
    '''
    function to 
    1. create tratified KFold
    2. cross validate the given classifier for each fold
    3. use the given eval_func
    4. the classifier requires fit and predict two functions

    noted that the StratifiedKFold function gives the location index, 
    not the DataFrame index
    '''
    
    results = []
    train_label = train_df[dep_var_name]
    skf = StratifiedKFold(train_label, fold_num, shuffle=True)

    for train, test in skf:
        kfold_train = train_df.iloc[train, :]
        kfold_test = train_df.iloc[test, :]
        kfold_test_label = train_label.iloc[test]
        classifier.fit(kfold_train, dep_var_name)
        scores = classifier.predict(kfold_test)
        result = eval_func(kfold_test_label, scores)
        results.append(result)
        
    return results


def MCC(ground_truth, scores):

    if isinstance(scores, pd.Series):
        scores = scores.values

    if isinstance(ground_truth, pd.Series):
        ground_truth = ground_truth.values

    tmp_ground_truth = np.copy(ground_truth)
    tmp_scores = np.copy(scores)
    ## convert to sk-learn format
    np.place(tmp_scores, tmp_scores == 0, -1)
    np.place(ground_truth, ground_truth == 0, -1)

    return matthews_corrcoef(tmp_ground_truth, tmp_scores)



def score_MCC(ground_truth, scores):
    '''
    assuming the model output is the probability of being default,
    then this probability can be used for ranking. Then using the fraction of
    default in validation data to assign the proper threshold to the prediction
    '''

    if isinstance(scores, pd.Series):
        scores = scores.values

    if isinstance(ground_truth, pd.Series):
        ground_truth = ground_truth.values

    tmp_ground_truth = np.copy(ground_truth)
    fault_frac = tmp_ground_truth.mean()
    #print 'score shape:', scores.shape, 
    print 'mean of groud truth:', fault_frac
    thres_value = np.percentile(scores, 100.*(1-fault_frac), axis=0)
    print 'threshold for preds:', thres_value
    binary_scores = scores > thres_value
    binary_scores = binary_scores.astype(int)
    ## convert to sk-learn format
    np.place(binary_scores, binary_scores==0, -1)
    np.place(tmp_ground_truth, tmp_ground_truth==0, -1)

    return matthews_corrcoef(tmp_ground_truth, binary_scores)



def create_validation_index(df, dep_var_name, valid_frac=0.2, to_shuffle=False, group_by_dep_var=False):
    '''
    function to create train/validation DataFrame index
    from a given DataFrame.
    '''
    valid_index = []
    train_index = []

    if group_by_dep_var:
        index_series = df[dep_var_name]
        grouped_index = index_series.groupby(index_series)

        for name, group in grouped_index:
            index_length = int(valid_frac * group.shape[0])
            valid_index.extend(group[0:index_length].index.tolist())
            train_index.extend(group[index_length:].index.tolist())

    train_index = df.index.to_series().sample(int(df.shape[0] * 1. * valid_frac), replace=False)
    valid_index = list(set(df.index.to_series()) - set(train_index))

    ## shuffle the training and test data in place
    if to_shuffle:
        shuffle(train_index)
        shuffle(valid_index)

    return  train_index, valid_index 

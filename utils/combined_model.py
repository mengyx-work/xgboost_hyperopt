from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import yaml, os, sys, time
from os.path import join, isfile
import cPickle as pickle
import abc
import pandas as pd
import numpy as np

## internal class
from wrapped_xgboost import xgboost_classifier
from models import BaseModel


class CombinedModel(BaseModel):

    def __init__(self, model_params):
        super(BaseModel, self).__init__()
        ## check the expected params for combined model
        expected_keys = ['raw_models_yaml_path', 'raw_models_yaml_file', 'project_path', 'models_yaml_file']
        for key in expected_keys:
            if key not in model_params:
                sys.exit('the expected key {} does not exist in params'.format(key))

        self.model_params = model_params

    
    @staticmethod
    def _initiate_model_by_type(model_type, model_params):
        ''' internal helper function to initiate and return
        proper model based on the 'model_type'
        '''
        if model_type == 'ExtraTree':
            model = ExtraTreeModel(model_params)

        if model_type == 'RandomForest':
            model = RandomForestModel(model_params)

        if model_type == 'Xgboost':
            ## directly use the xgboost_classifier to initiate the model
            ## pass the binary file name as a model param
            binary_file_name = model_params.pop(CombinedModel.xgb_binary_file_key, 'combined_model_xgboost_binary_file')
            use_weights      = model_params.pop('use_weights', False)
            model = xgboost_classifier(params = model_params, use_weights = use_weights, model_file = binary_file_name)

        return model


    
    @staticmethod
    def eval_mcc(y_true, y_prob):
        
        def mcc(tp, tn, fp, fn):
            sup = tp * tn - fp * fn
            inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            if inf==0:
                return 0
            else:
                return sup / np.sqrt(inf)

        idx = np.argsort(y_prob)
        y_true_sort = y_true[idx]
        n = y_true.shape[0]
        nump = 1.0 * np.sum(y_true) # number of positive
        numn = n - nump # number of negative
        tp = nump
        tn = 0.0
        fp = numn
        fn = 0.0
        best_mcc = 0.0
        mccs = np.zeros(n)
        for i in range(n):
            # all items with idx <= i are predicted negative while others are predicted positive
            if y_true_sort[i] == 1:
                tp -= 1.0
                fn += 1.0
            else:
                fp -= 1.0
                tn += 1.0
            new_mcc = mcc(tp, tn, fp, fn)
            mccs[i] = new_mcc
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_thres = y_prob[idx[i]]

        return float(best_mcc), float(best_thres)




    ## static method is called within a class method
    @classmethod
    def mcc_eval_func(cls, ground_truth, scores):
        if isinstance(scores, pd.Series):
            scores = scores.values
        if isinstance(ground_truth, pd.Series):
            ground_truth = ground_truth.values
        tmp_ground_truth = np.copy(ground_truth)
        tmp_scores = np.copy(scores)
        best_mcc, best_thres = cls.eval_mcc(tmp_ground_truth, tmp_scores)
        return best_mcc, best_thres



    @classmethod
    def build_cross_validate_models(cls, train_df, dep_var_name, model_dict, project_path, curr_max_model_index, fold_num=3):
        '''
        train_df: a Pandas DataFrame of training data
        model_dict: a dictionary of one set of model parameters
        fold_num: the number of folds to cross validate the model
        return:
        results: cross validation results
        thresholds: the thresholds for each trained model based on MCC score
        summary_dict: a collection of updated informations about trained model
        '''
        results, thresholds = [], []
        ## part of the models_dict
        summary_dict = {}
        train_label = train_df[dep_var_name]
        skf = StratifiedKFold(train_label, fold_num, shuffle=True)

        for train, test in skf:
            kfold_train = train_df.iloc[train, :]
            kfold_test = train_df.iloc[test, :]
            kfold_test_label = train_label.iloc[test]
            ## one model_dict for each trained model
            tmp_model_dict = model_dict.copy()
            stacking_dict = {}
            if tmp_model_dict['model_type'] != 'Xgboost':
                model_pickle_file = 'combinedModel_indexed_{}_{}_model_{}_folds.pkl'.format(curr_max_model_index, tmp_model_dict['model_type'], fold_num)
                tmp_model_dict['model_file'] = model_pickle_file
                model = cls._initiate_model_by_type(tmp_model_dict['model_type'], tmp_model_dict['model_params'])
                ## no copy of train, specific model will spawn a copy of training data
                model.fit(kfold_train, dep_var_name) 
                pickle.dump(model, open(os.path.join(project_path, model_pickle_file), 'wb'), -1)
            else:
                ## for xgboost model, a binary booster is saved insteead of the class object
                model_pickle_file = 'combinedModel_indexed_{}_{}_model_{}_folds'.format(curr_max_model_index, tmp_model_dict['model_type'], fold_num)
                tmp_model_dict['model_file'] = model_pickle_file
                tmp_model_dict['model_params'][CombinedModel.xgb_binary_file_key] = os.path.join(project_path, model_pickle_file)
                model = cls._initiate_model_by_type(tmp_model_dict['model_type'], tmp_model_dict['model_params'])
                model.fit(kfold_train, dep_var_name) 
 
            ## stacking models preparation
            index_df = pd.DataFrame(index=kfold_train.index)
            training_index_file = 'combinedModel_indexed_{}_{}_model_{}_folds_training_index.csv'.format(curr_max_model_index, tmp_model_dict['model_type'], fold_num)
            index_df.to_csv(os.path.join(project_path, training_index_file))
            stacking_dict['training_index_file'] = training_index_file
            stacking_dict['index_col_name'] = index_df.index.name

            scores = model.predict(kfold_test)
            result, threshold = cls.mcc_eval_func(kfold_test_label, scores)
            tmp_model_dict['model_threshold']   = threshold
            tmp_model_dict['result']            = result
            tmp_model_dict['stacking_dict']     = stacking_dict
            summary_dict[str(curr_max_model_index)] = tmp_model_dict
            curr_max_model_index += 1
            results.append(result)
            thresholds.append(threshold)

        return results, thresholds, summary_dict



    def cross_vlidate_fit(self, train, dep_var_name, fold_num=3):
        
        self.dep_var_name = dep_var_name

        with open(os.path.join(self.model_params['raw_models_yaml_path'], self.model_params['raw_models_yaml_file']), 'r') as yml_stream:
            models_dict = yaml.load(yml_stream)

        if not os.path.exists(self.model_params['project_path']):
            os.makedirs(self.model_params['project_path'])
        else:
            print 'the predict_path {} already exits, overwrite the contents...'.format(self.model_params['project_path'])

        curr_max_model_index = 0
        model_count = -1
        trained_yaml_file = join(self.model_params['project_path'], self.model_params['models_yaml_file'])

        ## loop through all the models in raw model yaml file
        for index, model_dict in models_dict.items():
            start_time = time.time()
            model_count += 1
            ## cross validate the model with training data
            results, thresholds, tmp_summary_dict = self.build_cross_validate_models(train, self.dep_var_name, model_dict, self.model_params['project_path'], curr_max_model_index, fold_num)
            curr_max_model_index = max([int(i) for i in tmp_summary_dict.keys()])
            curr_max_model_index += 1

            ## try to append tmp_summary_dict to yaml file of trained models
            if model_count == 0:
                if isfile(trained_yaml_file):
                    print 'warning! trained yaml file already exists, overwriting it...'
                summary_dict = tmp_summary_dict.copy()
            else:
                if not isfile(trained_yaml_file):
                    raise ValueError('failed to find trained yaml file {}'.format(trained_yaml_file))
                
                ## load the content from trained yaml file
                with open(trained_yaml_file, 'r') as yml_stream:
                    summary_dict = yaml.load(yml_stream)
                summary_dict.update(tmp_summary_dict)

            with open(trained_yaml_file, 'w') as yml_stream:
                yaml.dump(summary_dict, yml_stream, default_flow_style=False)
            print 'finished training {} model indexed {} combined model using {} minutes; total {} models out of {}'.format(model_dict['model_type'], index, round((time.time() -start_time)/60, 1), model_count+1, len(models_dict))




    def fit(self, train, dep_var_name, append_models=False):

        '''fit function for combined model
        store the dep_var_nme with the combined model. 
        because the Xgboost model can not be directly, 
        pickled, only a binary booster is saved.
        '''

        self.dep_var_name = dep_var_name

        with open(os.path.join(self.model_params['raw_models_yaml_path'], self.model_params['raw_models_yaml_file']), 'r') as yml_stream:
            models_dict = yaml.load(yml_stream)

        if not os.path.exists(self.model_params['project_path']):
            if append_models:
                raise ValueError('the project_path does not exists in append_models...')
            os.makedirs(self.model_params['project_path'])
        else:
            if not append_models:
                print 'the predict_path {} already exits, overwrite the contents...'.format(self.model_params['project_path'])

        ## param prepared for Kaggle Bosch
        mean_faulted_rate = np.mean(train[self.dep_var_name])

        if append_models:
            with open(os.path.join(self.model_params['project_path'], self.model_params['models_yaml_file']), 'r') as yml_stream:
                curr_models_dict = yaml.load(yml_stream)
            curr_max_model_index = max([int(i) for i in curr_models_dict.keys()]) + 1

        for index, model_dict in models_dict.items():
            if append_models:
                index = index + curr_max_model_index

            if model_dict['model_type'] != 'Xgboost':
                model_pickle_file = 'combinedModel_indexed_{}_{}_model.pkl'.format(index, model_dict['model_type'])
                model_dict['model_file'] = model_pickle_file
                model = self._initiate_model_by_type(model_dict['model_type'], model_dict['model_params'])
                ## no copy of train, specific model will spawn a copy of training data
                model.fit(train, self.dep_var_name) 
                pickle.dump(model, open(os.path.join(self.model_params['project_path'], model_pickle_file), 'wb'), -1)
            else:
                ## for xgboost model, a binary booster is saved insteead of the class object
                model_pickle_file = 'combinedModel_indexed_{}_{}_model'.format(index, model_dict['model_type'])
                model_dict['model_file'] = model_pickle_file
                model_dict['model_params'][CombinedModel.xgb_binary_file_key] = os.path.join(self.model_params['project_path'], model_pickle_file)
                model = self._initiate_model_by_type(model_dict['model_type'], model_dict['model_params'])
                model.fit(train, self.dep_var_name) 

            ## the same date share this param across different models
            model_dict['fault_rate'] = float(mean_faulted_rate)

            if append_models:
                curr_models_dict[index] = model_dict
            print 'finished training {} model indexed {} from combined model'.format(model_dict['model_type'], index)

        with open(os.path.join(self.model_params['project_path'], self.model_params['models_yaml_file']), 'w') as yml_stream:
            if append_models:
                yaml.dump(curr_models_dict, yml_stream, default_flow_style=False)
            else:
                yaml.dump(models_dict, yml_stream, default_flow_style=False)



    @staticmethod
    def convert_scores2binary(scores, thres, is_thres_pos=True):
        if not isinstance(scores, pd.Series):
            scores = pd.Series(scores)
        tmp_scores = scores.copy()
        tmp_scores[scores < thres] = 0
        tmp_scores[scores > thres] = 1
        if is_thres_pos:
            tmp_scores[scores == thres] = 1
        else:
            tmp_scores[scores == thres] = 0
        tmp_scores = tmp_scores.astype(int)
        return tmp_scores


    def get_MCC_results_fromStackedBoschModel(self, data, dep_var_name, res_col_name='res'):
        res_df = self.stacking_model_result(data, dep_var_name, res_col_name)
        res_df[dep_var_name] = data[dep_var_name]
        return self.mcc_eval_func(res_df[dep_var_name], res_df[res_col_name])



    def stacking_model_result(self, data, dep_var_name, res_col_name='res'):

        with open(os.path.join(self.model_params['project_path'], self.model_params['models_yaml_file']), 'r') as yml_stream:
            models_dict = yaml.load(yml_stream)

        stacked_df = pd.DataFrame(index=data.index)
        stacked_df[res_col_name] = np.nan

        if dep_var_name in data.columns:
            tmp_data = data.drop(dep_var_name, axis=1)
      
        for index, model_dict in models_dict.items():
            model_pickle_file = model_dict['model_file']

            if model_dict['model_type'] != 'Xgboost':
                model = pickle.load(open(os.path.join(self.model_params['project_path'], model_pickle_file), 'rb'))
            else:
                ## initiate the model without specifying the dep_var_name
                model = xgboost_classifier()
                model.load_model_from_file(os.path.join(self.model_params['project_path'], model_pickle_file))
                
            stacking_dict = model_dict['stacking_dict']
            train_index = pd.read_csv(os.path.join(self.model_params['project_path'], stacking_dict['training_index_file']), index_col=stacking_dict['index_col_name'])
            predict_index = data.index.difference(train_index.index)
            if not all(stacked_df.ix[predict_index, res_col_name].isnull()):
                raise ValueError('some elements are already predicted in stacking models')
            stacked_df.ix[predict_index, res_col_name] = model.predict(tmp_data.ix[predict_index]) 

        return stacked_df


    def predict(self, data, score_conversion_type='B', dep_var_name=None):
        '''predict function for combined model
        Results from each model is one column of returned DataFrame
        If data contains the column with 'dep_var_name', this function creates 
        new DataFrame without 'dep_var_name' column
        '''
        with open(os.path.join(self.model_params['project_path'], self.model_params['models_yaml_file']), 'r') as yml_stream:
            models_dict = yaml.load(yml_stream)

        pred_df = pd.DataFrame()
        col_index_names = []

        if dep_var_name is not None:
            tmp_data = data.drop(dep_var_name, axis=1)
        else:
            print 'warning, no label column removal in predict...'
            tmp_data = data
       
        for index, model_dict in models_dict.items():
            model_pickle_file = model_dict['model_file']
            col_index_names.append(str(index))

            if model_dict['model_type'] != 'Xgboost':
                model = pickle.load(open(os.path.join(self.model_params['project_path'], model_pickle_file), 'rb'))
            else:
                ## initiate the model without specifying the dep_var_name
                model = xgboost_classifier()
                model.load_model_from_file(os.path.join(self.model_params['project_path'], model_pickle_file))
                
            ## create one columns of result for one model
            column_name = 'model_{}_index_{}'.format(model_dict['model_type'], index)
            scores = model.predict(tmp_data) 

            if score_conversion_type =='A':
                ## scores conversion type A:
                ## for each model, scores are directly converted
                ## into binary results
                model_thres = model_dict['model_threshold']
                pred_df[column_name] = self.convert_scores2binary(scores, model_thres, is_thres_pos = False)
            elif score_conversion_type == 'B':
                ## scores conversion type B
                ## for each model, scores are converted into rank
                mean_faulted_rate = model_dict['fault_rate']
                pred_df[column_name] = pd.Series(scores).rank()
            else:
                ## generic Score conversion type C
                ## traditional way to collect data from each model
                pred_df[column_name] = scores

        if score_conversion_type == 'A':
            ## follow conversion type A
            ## a majority vote is used to calculate the final result
            mean_preds = pred_df.mean(axis=1)
            mean_result = self.convert_scores2binary(mean_preds, 0.5, is_thres_pos = False)
            return mean_result
        elif score_conversion_type == 'B':
            ## Conversion type B
            ## aggregate the predictions from different models
            result = pred_df.sum(axis=1)
            pred_data_index =data.index
            result.index = pred_data_index
            result.sort_values(inplace = True)
            print 'prediction using the mean faulted rate:', mean_faulted_rate
            thres_index = int(mean_faulted_rate * len(pred_data_index))
            result[:-thres_index] = 0
            result[-thres_index:] = 1
            ## align the results with input test data using index
            result = result.ix[pred_data_index]
            return result
        else: 
            ## aggregate on each row and return the sum
            pred_df.index = data.index
            pred_df.columns = col_index_names
            #result = pred_df.sum(axis=1)
            return pred_df




from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import yaml, os, sys, time
import cPickle as pickle
import abc
import pandas as pd
import numpy as np
## internal class
from wrapped_xgboost import xgboost_classifier


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def predict(self, data):
        pass


class CombinedModel(BaseModel):

    ## key for the xgboost binary file 
    xgb_binary_file_key = 'xgb_binary_file_name'

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
        '''
        helper function to initiate the 
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
            curr_max_model_index += 1
            if tmp_model_dict['model_type'] != 'Xgboost':
                model_pickle_file = 'combinedModel_indexed_{}_{}_model_{}_folds_eval.pkl'.format(curr_max_model_index, tmp_model_dict['model_type'], fold_num)
                tmp_model_dict['model_file'] = model_pickle_file
                model = cls._initiate_model_by_type(tmp_model_dict['model_type'], tmp_model_dict['model_params'])
                ## no copy of train, specific model will spawn a copy of training data
                model.fit(kfold_train, dep_var_name) 
                pickle.dump(model, open(os.path.join(project_path, model_pickle_file), 'wb'), -1)
            else:
                ## for xgboost model, a binary booster is saved insteead of the class object
                model_pickle_file = 'combinedModel_indexed_{}_{}_model_{}_folds_eval'.format(curr_max_model_index, tmp_model_dict['model_type'], fold_num)
                tmp_model_dict['model_file'] = model_pickle_file
                tmp_model_dict['model_params'][CombinedModel.xgb_binary_file_key] = os.path.join(project_path, model_pickle_file)
                model = cls._initiate_model_by_type(tmp_model_dict['model_type'], tmp_model_dict['model_params'])
                model.fit(kfold_train, dep_var_name) 
 
            scores = model.predict(kfold_test)
            result, threshold = cls.mcc_eval_func(kfold_test_label, scores)
            tmp_model_dict['model_threshold']   = threshold
            tmp_model_dict['result']            = result
            summary_dict[str(curr_max_model_index)] = tmp_model_dict
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

        summary_dict = {}
        curr_max_model_index = -1
        for index, model_dict in models_dict.items():
            results, thresholds, tmp_summary_dict = self.build_cross_validate_models(train, self.dep_var_name, model_dict, self.model_params['project_path'], curr_max_model_index, 2)
            curr_max_model_index = max([int(i) for i in tmp_summary_dict.keys()])
            summary_dict.update(tmp_summary_dict)
            print 'finished training {} model indexed {} from combined model'.format(model_dict['model_type'], index)


        with open(os.path.join(self.model_params['project_path'], self.model_params['models_yaml_file']), 'w') as yml_stream:
            yaml.dump(summary_dict, yml_stream, default_flow_style=False)





    def fit(self, train, dep_var_name, append_models=False):

        ## store the dep_var_nme on the combined model.
        ## because the Xgboost model can not be directly
        ## pickled, only a binary booster is saved.

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



    def predict(self, data, score_conversion_type = 'B'):
        '''
        Prediction results from each model is one column of returned DataFrame
        If data contains the column with 'dep_var_name', this functions create 
        an additioanl column of original 'dep_var_name'
        '''
        with open(os.path.join(self.model_params['project_path'], self.model_params['models_yaml_file']), 'r') as yml_stream:
            models_dict = yaml.load(yml_stream)

        pred_df = pd.DataFrame()
       
        for index, model_dict in models_dict.items():
            model_pickle_file = model_dict['model_file']

            if model_dict['model_type'] != 'Xgboost':
                model = pickle.load(open(os.path.join(self.model_params['project_path'], model_pickle_file), 'rb'))
            else:
                ## initiate the model without specifying the dep_var_name
                model = xgboost_classifier()
                model.load_model_from_file(os.path.join(self.model_params['project_path'], model_pickle_file))
                
            ## create one columns of result for one model
            column_name = 'model_{}_index_{}'.format(model_dict['model_type'], index)
            scores = model.predict(data) 

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
            result = pred_df.sum(axis=1)
            return result




class XgboostModel(BaseModel):

    def __init__(self, model_params):
        ## extract the use_weights param from the param dictionary
        ## also remove the 'use_weights' from the param dictionary
        use_weights = model_params.pop('use_weights', False)
        binary_model_file  = model_params.pop('binary_model_file', 'xgb_binary_model')
        super(BaseModel, self).__init__()
        self.model = xgboost_classifier(params = model_params, use_weights = use_weights, model_file = binary_model_file)


    def fit(self, data, dep_var_name=None):
        if dep_var_name is None:
            sys.exit('dep_var_name is needed for fit function.')
        else:
            self.dep_var_name = dep_var_name
        self.model.fit(data, self.dep_var_name)


    def predict(self, data):
        scores = self.model.predict(data)
        ## scores is a numpy array without index
        result = pd.Series(scores, index=data.index)
        return result




class ExtraTreeModel(BaseModel):

    def __init__(self, model_params):
        super(BaseModel, self).__init__()
        self.model = ExtraTreesClassifier(**model_params)


    def fit(self, data, dep_var_name=None):

        if dep_var_name is None:
            sys.exit('dep_var_name is needed for fit function.')
        else:
            self.dep_var_name = dep_var_name

        tmp_data = data.copy()
        data_label = tmp_data[self.dep_var_name].values
        tmp_data.drop(self.dep_var_name, axis=1, inplace=True)
        self.model.fit(tmp_data, data_label)


    def predict(self, data):

        if self.dep_var_name in data.columns:
            tmp_data = data.copy()
            tmp_data.drop(self.dep_var_name, axis=1, inplace=True)
        else:
            tmp_data = data

        scores = self.model.predict_proba(tmp_data)
        ## scores is a numpy array without index
        result = pd.Series(scores[:, 1], index=tmp_data.index)
        return result


class RandomForestModel(BaseModel):

    def __init__(self, model_params):
        super(BaseModel, self).__init__()
        self.model = RandomForestClassifier(**model_params)


    def fit(self, data, dep_var_name=None):

        if dep_var_name is None:
            sys.exit('dep_var_name is needed for fit function.')
        else:
            self.dep_var_name = dep_var_name

        tmp_data = data.copy()
        data_label = tmp_data[self.dep_var_name].values
        tmp_data.drop(self.dep_var_name, axis=1, inplace=True)
        self.model.fit(tmp_data, data_label)
        del tmp_data

    def predict(self, data):

        if self.dep_var_name in data.columns:
            tmp_data = data.copy()
            tmp_data.drop(self.dep_var_name, axis=1, inplace=True)
        else:
            tmp_data = data

        scores = self.model.predict_proba(tmp_data)
        result = pd.Series(scores[:, 1], index=tmp_data.index)
        return result



class LogisticRegressionModel(BaseModel):
    def __init__(self, model_params):
        super(BaseModel, self).__init__()
        self.model = LogisticRegression(**model_params)

    def fit(self, data, dep_var_name=None):

        if dep_var_name is None:
            sys.exit('dep_var_name is needed for fit function.')
        else:
            self.dep_var_name = dep_var_name

        tmp_data = data.copy()
        data_label = tmp_data[self.dep_var_name].values
        tmp_data.drop(self.dep_var_name, axis=1, inplace=True)
        self.model.fit(tmp_data, data_label)
        del tmp_data

    def predict(self, data):

        if self.dep_var_name in data.columns:
            tmp_data.drop(self.dep_var_name, axis=1, inplace=True)
            tmp_data = data.copy()
        else:
            tmp_data = data

        scores = self.model.predict_proba(tmp_data)
        result = pd.Series(scores[:, 1], index=tmp_data.index)
        return result

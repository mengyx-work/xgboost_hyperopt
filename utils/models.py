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
        expected_keys = ['raw_models_yaml_file', 'project_path', 'models_yaml_file']
        for key in expected_keys:
            if key not in model_params:
                sys.exit('the expected key {} does not exist in params'.format(key))

        self.model_params = model_params

    
    @classmethod
    def _initiate_model_by_type(self, model_type, model_params):
        '''
        helper function to initiate the 
        proper model based on the 'model_type'
        '''
        if model_type == 'ExtraTree':
            model = ExtraTreeModel(model_params)

        if model_type == 'RandomForest':
            model = RandomForestModel(model_params)

        if model_type == 'Xgboost':
            ## use the XgboostModel class
            #model = XgboostModel(model_params)

            ## directly use the xgboost_classifier
            binary_file_name = model_params.pop(CombinedModel.xgb_binary_file_key, 'combined_model_xgboost_binary_file')
            use_weights      = model_params.pop('use_weights', False)
            model = xgboost_classifier(params = model_params, use_weights = use_weights, model_file = binary_file_name)

        return model

               
    def fit(self, train, dep_var_name):

        ## store the dep_var_nme on the combined model.
        ## because the Xgboost model can not be directly
        ## pickled, only a binary booster is saved.

        self.dep_var_name = dep_var_name

        with open(os.path.join(self.model_params['raw_models_yaml_path'], self.model_params['raw_models_yaml_file']), 'r') as yml_stream:
            models_dict = yaml.load(yml_stream)

        if not os.path.exists(self.model_params['project_path']):
            os.makedirs(self.model_params['project_path'])
        else:
            print 'the predict_path {} already exits, overwrite the contents...'.format(self.model_params['project_path'])

        ## param prepared for Kaggle Bosch
        mean_faulted_rate = np.mean(train[self.dep_var_name])

        for index, model_dict in models_dict.items():

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

            ## Kaggle Bosch
            ## the same date share this param across different models
            model_dict['fault_rate'] = mean_faulted_rate
            print 'finished training {} model indexed {} from combined model'.format(model_dict['model_type'], index)

        with open(os.path.join(self.model_params['project_path'], self.model_params['models_yaml_file']), 'w') as yml_stream:
            yaml.dump(models_dict, yml_stream, default_flow_style=False)



    def predict(self, data):
        '''
        Prediction results from each model is one column of returned DataFrame
        If data contains the column with 'dep_var_name', this functions create 
        an additioanl column of original 'dep_var_name'
        '''
        with open(os.path.join(self.model_params['project_path'], self.model_params['models_yaml_file']), 'r') as yml_stream:
            models_dict = yaml.load(yml_stream)

        pred_df = pd.DataFrame()

        '''
        if self.dep_var_name in data.columns:
            ## add include the valid_label column in result
            #pred_df['valid_label'] = data[self.dep_var_name]
            valid_data = data.copy()
            valid_data.drop(self.dep_var_name, axis=1, inplace=True)
        else:
            valid_data = data
        '''

        for index, model_dict in models_dict.items():
            ## load the pickle file name
            model_pickle_file = model_dict['model_file']

            if model_dict['model_type'] != 'Xgboost':
                model = pickle.load(open(os.path.join(self.model_params['project_path'], model_pickle_file), 'rb'))
            else:
                model = xgboost_classifier(label_name = self.dep_var_name)
                model.load_model_from_file(os.path.join(self.model_params['project_path'], model_pickle_file))
                

            ## create one columns of result for one model
            column_name = 'model_{}_index_{}'.format(model_dict['model_type'], index)

            ## traditional way to collect data from each model
            #pred_df[column_name] = model.predict(data)

            ## Bosch
            ## convert scores into rank
            mean_faulted_rate = model_dict['fault_rate']
            scores = model.predict(data)
            #pred_df['score_' + column_name] = scores 
            pred_df[column_name] = pd.Series(scores).rank()

        #'''
        result = pred_df.sum(axis=1)
        pred_data_index =data.index
        result.index = pred_data_index
        result.sort_values(inplace = True)
        print 'prediction using the mean faulted rate:', mean_faulted_rate
        thres_index = int(mean_faulted_rate * len(pred_data_index))
        result[:-thres_index] = 0
        result[-thres_index:] = 1
        result = result.ix[pred_data_index]
        result.to_csv('tmp_results.csv')
        return result
        #'''
        

        '''
        pred_df.index = data.index
        pred_df['label'] = data[self.dep_var_name]
        pred_df.to_csv('tmp_pred_df.csv')
        result = pred_df.sum(axis=1)
        result.index = data.index
        return result
        '''




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

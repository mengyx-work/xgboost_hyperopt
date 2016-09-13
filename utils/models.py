from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import yaml, os, sys, time
import cPickle as pickle
import abc
import pandas as pd
import numpy as np


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def predict(self, data):
        pass


class CombinedModel(BaseModel):
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

        return model

               
    def fit(self, train, dep_var_name):
        self.dep_var_name = dep_var_name
        with open(os.path.join(self.model_params['project_path'], self.model_params['raw_models_yaml_file']), 'r') as yml_stream:
            models_dict = yaml.load(yml_stream)

        for index, model_dict in models_dict.items():
            model = self._initiate_model_by_type(model_dict['model_type'], model_dict['model_params'])
            ## no copy of train, specific model will spawn a copy of training data
            model.fit(train, self.dep_var_name) 
            print 'finished training model indexed {} from combined model'.format(index)
            model_pickle_file = 'indexed_{}_{}_model.pkl'.format(index, model_dict['model_type'])
            pickle.dump(model, open(os.path.join(self.model_params['project_path'], model_pickle_file), 'wb'), -1)
            model_dict['model_file'] = model_pickle_file

        with open(os.path.join(self.model_params['project_path'], self.model_params['models_yaml_file']), 'w') as yml_stream:
            yaml.dump(models_dict, yml_stream)



    def predict(self, data):
        '''
        Prediction results from each model is one column of returned DataFrame
        If data contains the column with 'dep_var_name', this functions create 
        an additioanl column of original 'dep_var_name'
        '''
        with open(os.path.join(self.model_params['project_path'], self.model_params['models_yaml_file']), 'r') as yml_stream:
            models_dict = yaml.load(yml_stream)

        pred_df = pd.DataFrame()

        if self.dep_var_name in data.columns:
            ## add include the valid_label column in result
            #pred_df['valid_label'] = data[self.dep_var_name]
            valid_data = data.copy()
            valid_data.drop(self.dep_var_name, axis=1, inplace=True)
        else:
            valid_data = data

        for index, model_dict in models_dict.items():
            model_pickle_file = model_dict['model_file']
            model = pickle.load(open(os.path.join(self.model_params['project_path'], model_pickle_file), 'rb'))
            column_name = 'model_{}_index_{}'.format(model_dict['model_type'], index)
            pred_df[column_name] = model.predict(valid_data)

        result = pred_df.sum(axis=1)
        result.index = valid_data.index
        return result




class XgboostModel(BaseModel):
    def __init__(self, model_params, use_weights=False):
        super(BaseModel, self).__init__()
        self.model = xgboost_classifier(params = model_params, use_weights = use_weights)

    def fit(self, data, dep_var_name=None):
        if dep_var_name is None:
            sys.exit('dep_var_name is needed for fit function.')
        else:
            self.dep_var_name = dep_var_name
        self.model.fit(data, data_label)

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
            tmp_data.drop(self.dep_var_name, axis=1, inplace=True)
            tmp_data = data.copy()
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
            tmp_data.drop(self.dep_var_name, axis=1, inplace=True)
            tmp_data = data.copy()
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

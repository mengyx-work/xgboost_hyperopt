from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import yaml, os, sys, time
import cPickle as pickle
import abc
import pandas as pd
import numpy as np

'''
raw_models_yaml_file = 'raw_combined_models.yml'
project_path = '/mnt/home/ymm/kaggle/compete/current/combined_model'
#project_path = '/Users/ymm/Google_Drive/Kaggle/compete/current/combined_model'
trained_model_yaml_file = 'trained_combined_model.yml'

dep_var_name = 'Response'
project_yml_path = '/mnt/home/ymm/kaggle/compete/current'
data_path = '/mnt/home/ymm/kaggle/bosch_data/bosch_complete_processed_data'
data_yaml_file = 'bosch_processed_data_dict.yml'
    
train = load_processed_bosch_data(data_path, project_yml_path, data_yaml_file, data_index='0')
'''


def predict_combined_model(data, project_path, models_yaml_file, eval_func, dep_var_name=None): 

    with open(os.path.join(project_path, models_yaml_file), 'r') as yml_stream:
        models_dict = yaml.load(yml_stream)

    pred_df = pd.DataFrame()
    valid_data = data.copy()

    if dep_var_name is not None:
        pred_df['valid_label'] = data[dep_var_name]
        valid_data.drop(dep_var_name, axis=1, inplace=True)

    for index, model_dict in models_dict.items() :
        model_pickle_file = model_dict['model_file']
        model = pickle.load(open(os.path.join(project_path, model_pickle_file), 'rb'))
        column_name = 'model_{}_index_{}'.format(model_dict['model_type'], index)
        pred_df[column_name] = model.predict(valid_data)

    return pred_df

  

def train_combined_model(train, dep_var_name, raw_models_yaml_file, project_path, trained_model_yaml_file): 
    
    with open(os.path.join(project_path, raw_models_yaml_file), 'r') as yml_stream:
        models_dict = yaml.load(yml_stream)

    for index, model_dict in models_dict.items() :
        tmp_train = train.copy()
        train_label = train[dep_var_name]
        tmp_train.drop(dep_var_name, axis=1, inplace=True)
      
        if model_dict['model_type'] == 'ExtraTree':
            model = ExtraTreeModel(model_dict['model_params'])
        
        model.fit(tmp_train, train_label)
        print 'finished training model indexed {} from combined model'.format(index)
        model_pickle_file = 'indexed_{}_{}_model.pkl'.format(index, model_dict['model_type'])
        pickle.dump(model, open(model_pickle_file, 'wb'), -1)
        model_dict['model_file'] = model_pickle_file

    with open(os.path.join(project_path, trained_model_yaml_file), 'w') as yml_stream:
        yaml.dump(models_dict, yml_stream)


class BaseModel(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def fit(self, data):
        pass
    
    @abc.abstractmethod
    def predict(self, data):
        pass
    

class ExtraTreeModel(BaseModel):
    
    def __init__(self, model_params):
        super(BaseModel, self).__init__()
        self.model = ExtraTreesClassifier(**model_params)
        
    def fit(self, data, label):
        if isinstance(label, pd.Series):
            label = label.values

        self.model.fit(data, label)
        
    def predict(self, data):
        scores = self.model.predict_proba(data)
        print 'score shape:', scores.shape
        return scores[:, 1]



class RandomForestModel(BaseModel):
    
    def __init__(self, model_params):
        super(BaseModel, self).__init__()
        self.model = RandomForestClassifier(**model_params)
        
    def fit(self, data, label):
        if isinstance(label, pd.Series):
            label = label.values
        self.model.fit(data, label)
        
    def predict(self, data):
        scores = self.model.predict_proba(data)
        return scores[:, 1]



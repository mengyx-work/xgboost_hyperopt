from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
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



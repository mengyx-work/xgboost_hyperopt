import numpy as np
import pandas as pd
import xgboost as xgb
import sys
import os.path
from sklearn import metrics
from sklearn import datasets

class utils(object):

    # function to handle the data source, either a pd.data frame or a file path
    @staticmethod
    def _read_data(data_source):

        if isinstance(data_source, pd.DataFrame):
            return data_source

        elif isinstance(data_source, str):
            try:
                if os.path.isfile(file_path):
                    ValueError('failed to find data file')
                    sys.exit(0)
                # try to read the data from file
                data = pd.read_csv(data_source)
            except IOError:
                print 'cannot open', data_source, ' as a regular data source.'
                sys.exit(0)

            return data_source

        else:
            print 'unknown data_source is used.'
            sys.exit(0)

    @staticmethod
    def get_mockup_data(dep_var_name):

        # get the standard training data
        digits = datasets.load_digits(2)
        X_digits = digits.data
        y_digits = digits.target

        data = pd.DataFrame(np.column_stack((X_digits, y_digits)))

        # create dep_var in the data
        col_names = data.columns.tolist()
        col_names[len(col_names) - 1] = dep_var_name
        data.columns = col_names

        return data

    @classmethod
    def create_validation_data(self, file_path, valid_frac = 0.2, dep_var_name = 'dep_var'):

        train_data = utils._read_data(file_path)
        grouped_training_data = train_data.groupby(dep_var_name)
        valid_data = pd.DataFrame()
        train_data = pd.DataFrame()

        # split the data by dep_var levels
        for name, group in grouped_training_data:
            index_length = int(valid_frac * group.shape[0])
            #print group.shape, 'the index_length: ', index_length
            valid_data = valid_data.append(group[0:index_length])
            train_data = train_data.append(group[index_length:])

        # shuffle the training and test data by index
        valid_data = valid_data.reindex(np.random.permutation(valid_data.index))
        train_data = train_data.reindex(np.random.permutation(train_data.index))

        return train_data, valid_data

    @classmethod
    def convert_xgboost_data(self, train_data, dep_var_name='dep_var'):
        labels    = train_data[dep_var_name]
        X         = train_data.drop(dep_var_name, axis=1)
        xgb_train = xgb.DMatrix(np.array(X), label = np.array(labels), missing = np.NaN)

        return xgb_train

    @classmethod
    def calculat_AUC(self, labels, pred_res, pos_label = 1):
        fpr, tpr, thresholds = metrics.roc_curve(labels, pred_res, pos_label = pos_label)
        auc = metrics.auc(fpr, tpr)
        return auc

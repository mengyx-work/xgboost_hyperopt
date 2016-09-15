import os, sys, time
import yaml

models_yaml_file = 'raw_combined_models.yml'
#project_path = '/Users/ymm/Google_Drive/Kaggle/compete/current/combined_model'
#project_path = '/home/ymm/kaggle/compete/current'
#project_path = '/mnt/home/ymm/kaggle/compete/current/combined_model'
#project_path = '/mnt/home/ymm/kaggle/compete/current/tmp'
project_path = './'

if not os.path.exists(project_path):
    os.makedirs(project_path)

models = {}
model_index = 0

## for different random seed
def auto_fill_model_dicts(models, const_params, list_param_dict, model_type, model_index=None):
    if model_index is None:
        model_index = -1

    for key, list_param in list_param_dict.items():
        for param in list_param:
            model_params = const_params.copy()
            model_dict = {}
            model_index += 1
            model_params[key] = param
            model_dict['model_params'] = model_params
            model_dict['model_type'] = model_type
            print 'build a {} model using params: {}'.format(model_type, model_params)
            models[model_index] = model_dict

'''
const_params = {'random_state' : 9999, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
list_param_dict = {}
#list_param_dict['random_state'] = range(0, 10000, 100)
list_param_dict['random_state'] = range(0, 3)
auto_fill_model_dicts(models, const_params, list_param_dict, 'ExtraTree')
'''

'''
model_dict = {}
params = {}
params["eta"]                      = 0.0075
params["subsample"]                = 0.8
params["colsample_bytree"]         = 0.8
params["num_round"]                = 15
params["max_depth"]                = 5
params["gamma"]                    = 0
params["metrics"]                  = 'auc'
params['eval_metric']              = 'auc'
params["seed"]                     = 100
## whether to use weights
params['use_weights']              = True
params["val"]                      = False
#params["early_stopping_ratio"]     = 0.2

model_dict['model_type'] = 'Xgboost'
model_dict['model_params'] = params
models[model_index] = model_dict
'''

#'''
## ExtraTree Model
model_index += 1
model_dict = {}
model_dict['model_type'] = 'ExtraTree'
model_dict['model_params'] = {'random_state' : 100, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
models[model_index] = model_dict

## RandomForest Model
model_index += 1
model_dict = {}
model_dict['model_type'] = 'RandomForest'
model_dict['model_params'] = {'random_state' : 100, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
models[model_index] = model_dict
#'''



with open(os.path.join(project_path, models_yaml_file), 'w') as yml_stream:
    yaml.dump(models, yml_stream, default_flow_style=False)


'''
model_index = 0
model_dict = {}
model_dict['model_type'] = 'ExtraTree'
model_dict['model_params'] = {'random_state' : 0, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
models[model_index] = model_dict


model_index += 1
model_dict = {}
model_dict['model_type'] = 'ExtraTree'
model_dict['model_params'] = {'random_state' : 100, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
models[model_index] = model_dict


model_index += 1
model_dict = {}
model_dict['model_type'] = 'ExtraTree'
model_dict['model_params'] = {'random_state' : 1000, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
models[model_index] = model_dict


model_index += 1
model_dict = {}
model_dict['model_type'] = 'ExtraTree'
model_dict['model_params'] = {'random_state' : 9999, 'n_estimators' : 500, 'max_depth' : 4, 'criterion' : 'entropy', 'n_jobs' : -1}
models[model_index] = model_dict
'''

import sys
sys.path.append('/home/ymm/kaggle/xgboost_hyperopt')
from utils.validation_tools import combine_prediction_results_for_combined_models

data_path               ='/home/ymm/kaggle/bosch/data_2_bins_cross_fit_xgb_models'
submission_sample_file  = 'sample_submission.csv'
submission_sample_path  = '/mnt/home/ymm/bosch'
combine_prediction_results_for_combined_models(data_path, submission_sample_path, submission_sample_file, index_col_name = 'Id', res_col_name = 'Response') 


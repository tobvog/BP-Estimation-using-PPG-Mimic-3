import numpy as np
import os

from sklearn.model_selection import KFold
from Classical_ML import Classical_ML

#%% Load Data
path= "E:/Uni/Master BMIT/Programmierprojekt/feat2/"
files = np.array(os.listdir(path+"ground_truth/nn/")) 
y = np.array([np.load(path+"ground_truth/nn/"+subject, allow_pickle=True) for subject in files], dtype=object)
n_splits = 10

if __name__ == "__main__":
    
    ml = Classical_ML()
    kfold = KFold(n_splits=n_splits)
    kfold.get_n_splits(files)
    all_mae_sbp, all_mae_dbp = [], []
    nr_fold = 1
    
    for train_index, test_index in kfold.split(files):
        print("Number Fold: ", [nr_fold], " of ", [n_splits])
        nr_fold += 1
        
        mae_sbp, mae_dbp = ml.dummy(y, train_index, test_index)
        
        all_mae_sbp.append(mae_sbp)
        all_mae_dbp.append(mae_dbp)
        
    mean_mae_sbp = np.mean(all_mae_sbp)
    mean_mae_dbp = np.mean(all_mae_dbp)
    print("Mean MAE of SBP: ", mean_mae_sbp)
    print("Mean MAE of DBP: ", mean_mae_dbp)
import numpy as np
import os

from sklearn.model_selection import KFold
from Classical_ML import Classical_ML

#%% Load Data
path= "E:/Uni/Master BMIT/Programmierprojekt/feat/"
files = np.array(os.listdir(path+"ground_truth/ml/")) 
y = np.array([np.load(path+"ground_truth/ml/"+subject, allow_pickle=True) for subject in files], dtype=object)
n_splits = 10
# label=0 -> sbp; label=1 -> dbp
label = 0

if __name__ == "__main__":
    ml = Classical_ML()
    kfold = KFold(n_splits=n_splits)
    kfold.get_n_splits(files)
    all_mae = []
    nr_fold = 1
    
    for train_index, test_index in kfold.split(files):
        print("Number Fold: ", [nr_fold], " of ", [n_splits])
        nr_fold += 1
        
        mae = ml.rfregression(files, y, train_index, test_index, path, batch_size=1, label=label)
        
        all_mae.append(mae)
        
    mean_mae = np.mean(all_mae)
    print("Mean MAE: ", mean_mae)

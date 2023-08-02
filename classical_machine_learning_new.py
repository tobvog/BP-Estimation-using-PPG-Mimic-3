
"""
Skript for the Classical Machine Learning Part of the Paper ....
"""
#%% Imports
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
#%%

def spec_flatten(x_in=None, y_in=None, only_y=False):      
    x_out, y_out = [], []
    for sub in range(0, len(y_in)):
        for cyc in range(0, len(y_in[sub])):
            if only_y==False:    
                temp_x = x_in[sub][cyc]
            temp_y = y_in[sub][cyc]
            if np.isnan(temp_y[0])==False and np.isnan(temp_y[1])==False:
                if only_y==False:
                    x_out.append(temp_x)
                y_out.append(temp_y)
    if only_y==False:
        return np.array(x_out, dtype=object), np.array(y_out)
    else:
        return np.array(y_out)
    

def spec_flatten_y(y_in):
    y_out = []
    for sub in range(0, len(y_in)):
        for cyc in range(0, len(y_in[sub])):
            temp_y = y_in[sub][cyc]
            if np.isnan(temp_y[0])==False and np.isnan(temp_y[1])==False:
                y_out.append(temp_y)
                
    return np.array(y_out)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
def dummy(y, train_index, test_index):
    y_train, y_test = y[train_index], y[test_index]
    y_train = spec_flatten(y_in=y_train, only_y=True)
    y_test = spec_flatten(y_in=y_test, only_y=True)

    y_pred_sbp = np.full(len(y_test), np.mean(y_train[:,0])) 
    y_pred_dbp = np.full(len(y_test), np.mean(y_train[:,1])) 

    y_test = y[test_index]   
    y_test = spec_flatten_y(y_test, only_y=True)
    y_pred_sbp = np.full(len(y_test), np.mean(y_train[:,0]))  
    # if len(y_test) == 0:
    #     continue
    for mini_batch in batch(train_index, batch_size):
        print("Batch nr: "+str(nr_batch)+" of "+str(nr_batches))
        nr_batch += 1
        y_train = y[mini_batch]
        y_train = spec_flatten(y_in=y_train, only_y=True)
        y_train_sbp = y_train[:,0]   
        y_train_dbp = y_train[:,1] 
   
        y_test = y[test_index]   
        y_test = spec_flatten_y(y_test)
        if len(y_test) == 0:
            continue
        y_test = y_test[:,label]
   
        y_pred = np.full(len(y_test), np.mean(y_train))  
        mae = mean_absolute_error(y_test, y_pred)   
        all_mae.append(mae)
    
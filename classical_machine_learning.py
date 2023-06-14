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

def spec_flatten(x_in, y_in):      
    x_out, y_out = [], []
    for sub in range(0, len(x_in)):
        for cyc in range(0, len(x_in[sub])):
            temp_x = x_in[sub][cyc]
            temp_y = y_in[sub][cyc]
            if np.isnan(temp_y[0])==False and np.isnan(temp_y[1])==False:
                x_out.append(temp_x)
                y_out.append(temp_y)
    
    return np.array(x_out, dtype=object), np.array(y_out)

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

#%% Load Data
# path = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/feat/"
path= "E:/Uni/Master BMIT/Programmierprojekt/feat/"
path2 = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/"
files = np.array(os.listdir(path2+"feature")) 

#x = np.array([np.load(path2+"feature/"+subject, allow_pickle=True) for subject in files], dtype=object)
#x = np.array([np.load(path+"feature/"+subject, allow_pickle=True) for subject in files], dtype=object)
y = np.array([np.load(path+"ground_truth/ml/"+subject, allow_pickle=True) for subject in files], dtype=object)
batch_size = 128

#%% Settings #######################
pers = False
dummy = False
# label = 0 SBP
# label = 1 DBP
label = 0
####################################
#clf = RandomForestRegressor(warm_start=True)
n_splits = 10
kfold = KFold(n_splits=n_splits)
kfold.get_n_splits(files)

all_mae = []
#all_mae = list(np.load("C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/testing/dummy_mae_sbp.npy"))
nr_fold = 1
for train_index, test_index in kfold.split(files):
    #if test_index < len(all_mae):
    #    continue
    #clf = RandomForestRegressor(warm_start=True, verbose=2, n_jobs=-1)
    #loo = LeaveOneOut()
    
    if pers==True and dummy==True:
        print("The variables _pers_ and _dummy_ must not both be True!")
        break
    
    print("Number Fold: ", [nr_fold], " of ", [n_splits])
    '''
    y_train = y[train_index]
    y_test = y[test_index]   
    y_train = spec_flatten_y(y_train)
    y_test = spec_flatten_y(y_test)
    
    #if len(y_test) == 0:
    #    continue
    
    y_test = y_test[:,label]
    y_train = y_train[:,label]
    '''
########################################### 
###########################################   
###########################################    
    if pers==False and dummy==True:  

        y_train = y[train_index]
        y_test = y[test_index]   
        y_train = spec_flatten_y(y_train)
        y_test = spec_flatten_y(y_test)
        if len(y_test) == 0:
            continue
        y_test = y_test[:,label]
        y_train = y_train[:,label]
        

        
        y_pred = np.full(len(y_test), np.mean(y_train))
        
        mae = mean_absolute_error(y_test, y_pred)   
        all_mae.append(mae)
########################################### 
###########################################   
###########################################              
    elif pers==False and dummy==False: 
        clf = RandomForestRegressor(verbose=2, n_jobs=-1, warm_start=True)
        
        #x_test = np.array([np.load(path2+"feature/"+subject, allow_pickle=True) for subject in files[test_index]], dtype=object)
        # 
        # x_train, y_train = x[train_index], y[train_index]
        
        # x_test, y_test = spec_flatten(x_test, y_test)
        # x_train, y_train = spec_flatten(x_train, y_train)
        # y_test = y_test[:,label]
        nr_batch = 1
        for mini_batch in batch(train_index, batch_size):
            print("Batch nr: "+str(nr_batch)+" of 8")
            #x_train = np.load(path+"feature/ml/"+files[mini_batch], allow_pickle=True)
            x_train = np.array([np.load(path2+"feature/"+subject, allow_pickle=True) for subject in files[mini_batch]], dtype=object)
            y_train = y[mini_batch]
            x_train, y_train = spec_flatten(x_train, y_train)
            y_train = y_train[:,label]
           
            clf.fit(x_train, y_train)
            clf.n_estimators += 1
        
        x_test = np.array([np.load(path2+"feature/"+subject, allow_pickle=True) for subject in files[test_index]], dtype=object)
        y_test = y[test_index]
        x_test, y_test = spec_flatten(x_test, y_test)
        y_test = y_test[:,label]
        y_pred = clf.predict(x_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        
        all_mae.append(mae)
########################################### 
###########################################   
########################################### 
    elif pers==True and dummy==False:
        clf = RandomForestRegressor(warm_start=True, verbose=2, n_jobs=4)
        nr_cyc = int(len(x_test)*0.2)
        
        x_test = np.load(path+"feature/ml/"+files[test_index], allow_pickle=True) 
        y_test = y[test_index]
        x_test, y_test = spec_flatten(x_test, y_test)
        y_test = y_test[:,label]
        
        x_test = x_test[nr_cyc:]
        y_test = y_test[nr_cyc:]
        
        for mini_batch in batch(train_index, batch_size):
            x_train = np.load(path+"feature/ml/"+files[mini_batch], allow_pickle=True)
            y_train = y[mini_batch]
            x_train, y_train = spec_flatten(x_train, y_train)
            y_train = y_train[:,label]
            
            clf.fit(x_train, y_train)
        
        clf.fit(x_test[:nr_cyc], y_test[:nr_cyc])
        y_pred = clf.predict(x_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        
        all_mae.append(mae)

mean_mae = np.mean(all_mae)
print("Mean MAE: ", mean_mae)

#np.save("C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/testing/dummy_mae_sbp.npy", all_mae)
        








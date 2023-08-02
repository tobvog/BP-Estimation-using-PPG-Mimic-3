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

#%% Load Data
#path = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/feat_new/"
path= "E:/Uni/Master BMIT/Programmierprojekt/feat2/"
path2 = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/"
files = np.array(os.listdir(path+"ground_truth/nn/")) 

#x = np.array([np.load(path2+"feature/"+subject, allow_pickle=True) for subject in files], dtype=object)
#x = np.array([np.load(path+"feature/"+subject, allow_pickle=True) for subject in files], dtype=object)
y = np.array([np.load(path+"ground_truth/nn/"+subject, allow_pickle=True) for subject in files], dtype=object)
batch_size = 64
nr_batches = int(len(y)/batch_size)

#%% Settings #######################
pers = False
dummy = True
# label = 0 SBP
# label = 1 DBP
label = 0
####################################
n_splits = 10
kfold = KFold(n_splits=n_splits)
kfold.get_n_splits(files)

all_mae = []
nr_fold = 1
for train_index, test_index in kfold.split(files):
    
    if pers==True and dummy==True:
        print("The variables _pers_ and _dummy_ must not both be True!")
        break
    
    print("Number Fold: ", [nr_fold], " of ", [n_splits])
    nr_fold += 1
########################################### 
###########################################   
###########################################    
    if pers==False and dummy==True:  
        nr_batch = 1
        for mini_batch in batch(train_index, batch_size):
            print("Batch nr: "+str(nr_batch)+" of "+str(nr_batches))
            nr_batch += 1
            #x_train = np.array([np.load(path2+"feature/"+subject, allow_pickle=True) for subject in files[mini_batch]], dtype=object)
            y_train = y[mini_batch]
            #x_train, y_train = spec_flatten(x_train, y_train)
            y_train = spec_flatten(y_in=y_train, only_y=True)
            y_train = y_train[:,label]        
       
            y_test = y[test_index]   
            y_test = spec_flatten_y(y_test)
            if len(y_test) == 0:
                continue
            y_test = y_test[:,label]
       
            y_pred = np.full(len(y_test), np.mean(y_train))  
            mae = mean_absolute_error(y_test, y_pred)   
            all_mae.append(mae)
########################################### 
###########################################   
###########################################              
    elif pers==False and dummy==False: 
        clf = RandomForestRegressor(n_estimators=13, verbose=2, n_jobs=3, warm_start=True)
        
        nr_batch = 1
        for mini_batch in batch(train_index, batch_size):
            print("Batch nr: "+str(nr_batch)+" of "+str(nr_batches))
            nr_batch += 1
            x_train = np.array([np.load(path2+"feature/"+subject, allow_pickle=True) for subject in files[mini_batch]], dtype=object)
            y_train = y[mini_batch]
            x_train, y_train = spec_flatten(x_train, y_train)
            y_train = y_train[:,label]
           
            clf.fit(x_train, y_train)
            clf.n_estimators += 13
        
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
        clf = RandomForestRegressor(n_estimators=13, warm_start=True, verbose=2, n_jobs=4)
        '''
        x_test = np.load(path2+"feature/"+files[test_index], allow_pickle=True)
        y_test = y[test_index]
        x_test, y_test = spec_flatten(x_test, y_test)
        y_test = y_test[:,label]
        '''
        x_test = np.array([np.load(path2+"feature/"+subject, allow_pickle=True) for subject in files[test_index]], dtype=object)
        y_test = y[test_index]
        x_test, y_test = spec_flatten(x_test, y_test)
        y_test = y_test[:,label]

        nr_cyc = int(len(x_test) * 0.2)
        x_test = x_test[nr_cyc:]
        y_test = y_test[nr_cyc:]
        nr_batch = 1
        for mini_batch in batch(train_index, batch_size):
            print("Batch nr: "+str(nr_batch)+" of "+str(nr_batches))
            nr_batch += 1
            x_train = np.array([np.load(path2+"feature/"+subject, allow_pickle=True) for subject in files[mini_batch]], dtype=object)
            y_train = y[mini_batch]
            x_train, y_train = spec_flatten(x_train, y_train)
            y_train = y_train[:,label]
            
            clf.fit(x_train, y_train)
            clf.n_estimators += 13
            '''
            if nr_batch==nr_batches:
                clf.n_estimators += 85
            else:
                clf.n_estimators += 13
            '''


        clf.fit(x_test[:nr_cyc], y_test[:nr_cyc])
        y_pred = clf.predict(x_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        
        all_mae.append(mae)

mean_mae = np.mean(all_mae)
print("Mean MAE: ", mean_mae)

#np.save("C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/testing/dummy_mae_sbp.npy", all_mae)
        








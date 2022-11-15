"""
Skript for the Classical Machine Learning Part of the Paper ....
"""
#%% Imports
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
#%%
def spec_flatten(x_in, y_in):      
    x_out, y_out = [], []
    for sub in range(0, len(x_in)):
        print("Subject: "+str(sub+1))
        for cyc in range(0, len(x_in[sub])):
            temp_x = x_in[sub][cyc]
            temp_y = y_in[sub][cyc]
            #print("sub: "+str(sub)+" & cyc: "+str(cyc))
            #print(np.shape(temp_y))
            if np.isnan(temp_y[0])==False and np.isnan(temp_y[1])==False:
                x_out.append(temp_x)
                y_out.append(temp_y)
    
    return np.array(x_out, dtype=object), np.array(y_out)
#%% Load Data
#path = "" 
x = np.load("feature_test.npy", allow_pickle=True)
y = np.load("label_test.npy", allow_pickle=True)
#%% Settings
pers = True
dummy = False
# label = 0 SBP
# label = 1 DBP
label = 0
#%%
clf = RandomForestRegressor()
loo = LeaveOneOut()
loo.get_n_splits(x)

all_mae = []
for train_index, test_index in loo.split(x):
    if pers==True and dummy==True:
        print("The variables _pers_ and _dummy_ must not both be True!")
        break
            
    print("Learning Part: ", test_index, " of ", [len(x)])
    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]  
    x_train, y_train = spec_flatten(x_train, y_train)
    x_test, y_test = spec_flatten(x_test, y_test)
    
    if pers==False and dummy==True:
        y_pred = np.full(len(y_test), np.mean(y_train[:][label]))
        mae = mean_absolute_error(y_test[:, label], y_pred)   
        all_mae.append(mae)
        
    elif pers==True and dummy==False:
        nr_cyc = int(len(x_test)*0.2)
        x_train = np.concatenate((x_train, x_test[:nr_cyc]))
        y_train = np.concatenate((y_train, y_test[:nr_cyc]))     
        x_test = x_test[nr_cyc:]
        y_test = y_test[nr_cyc:]
        
        clf.fit(x_train, y_train[:, label])
        y_pred = clf.predict(x_test)
        
        mae = mean_absolute_error(y_test[:, label], y_pred)
        
        all_mae.append(mae)
    
    elif pers==False and dummy==False:      
        clf.fit(x_train, y_train[:, label])
        y_pred = clf.predict(x_test)
        
        mae = mean_absolute_error(y_test[:, label], y_pred)
        
        all_mae.append(mae)
        

mean_mae = np.mean(all_mae)
print("Mean MAE: ", mean_mae)




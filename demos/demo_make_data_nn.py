import numpy as np
import os
from ML_Preparing import ML_Preparing

#files = os.listdir("E:/Uni/Master BMIT/Programmierprojekt/feat/derivations/peaks/")  
    
#path_time = "E:/Uni/Master BMIT/Programmierprojekt/passed_subjects2/"
path_time = "D:/MIMICIII_Database/preprocessed_data_slapnicar/"
#path_peak = "E:/Uni/Master BMIT/Programmierprojekt/feat/derivations/peaks/"
path_peak = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/peaks/"
add_time = "preprocessed_data/"
files = os.listdir(path_peak)  
target_path = "D:/MIMICIII_Database/feat/"   


for no, sub in enumerate(files[:50]):
    print("Sub no "+str(no+1)+" of "+str(len(files)))
    
    peaks = np.load(path_peak+sub)
    data_time = np.load(path_time+sub, allow_pickle=True)
    
    ml_prep = ML_Preparing(pleth=data_time[0], abp=data_time[1], idx_peak=peaks)
    
    target, pleth_cyc = ml_prep.extract_sbp_dbp(window=5, pad=110, nn_epoch=True)
    
    ml_prep.pleth_cyc = pleth_cyc
    dev1, dev2 = ml_prep.derivation2()
    
    np.save(target_path+"derivations/dev0/"+sub, pleth_cyc)
    np.save(target_path+"derivations/dev1/"+sub, dev1)
    np.save(target_path+"derivations/dev2/"+sub, dev2)
    np.save(target_path+"ground_truth/nn/"+sub, target)


#%%
test = np.load("E:/Uni/Master BMIT/Programmierprojekt/feat2/derivations/dev2/"+files[0], allow_pickle=True)
    
test = np.array(test, dtype=float)   
    
for i in test:
    if len(i) == 624:
        continue
    print(len(i))
    
#%%
#path_main = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/feat_new/"
path_main = "D:/MIMICIII_Database/feat/"
files = os.listdir(path_main+"/derivations/dev0/")
for id_ in files:
    target = np.load(path_main+"ground_truth/nn/"+id_)
    nan_indices = np.isnan(target).any(axis=1)
    print(id_)
    if np.any(nan_indices):
        print("NaN values found in target data at indices:", np.where(nan_indices)[0])
        os.remove(path_main+"ground_truth/nn/"+id_)
        os.remove(path_main+"derivations/dev0/"+id_)
        os.remove(path_main+"derivations/dev1/"+id_)
        os.remove(path_main+"derivations/dev2/"+id_)
        




    
    
    
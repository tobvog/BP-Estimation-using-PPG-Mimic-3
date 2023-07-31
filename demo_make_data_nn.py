import numpy as np
import os
from ML_Preparing import ML_Preparing

files = os.listdir("E:/Uni/Master BMIT/Programmierprojekt/feat/derivations/peaks/")  
    
path_time = "E:/Uni/Master BMIT/Programmierprojekt/passed_subjects2/"
path_peak = "E:/Uni/Master BMIT/Programmierprojekt/feat/derivations/peaks/"
add_time = "preprocessed_data/"
files = os.listdir(path_peak)  
target_path = "E:/Uni/Master BMIT/Programmierprojekt/feat2/"   


for no, sub in enumerate(files):
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

ar = np.array([0,1,2,3,4,5])
ar = np.append(ar, [6,7,8,9])

    
    
    
    
    
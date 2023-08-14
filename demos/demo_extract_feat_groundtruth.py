import numpy as np
import os
from ML_Preparing import ML_Preparing

# Main path of the data
path_main = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/"
# Target path
target_path = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/feat/"
# Addable path elements
add_cyc = "cycled_data/"  
add_time = "preprocessed_data/"

ids = os.listdir(path_main+add_time) 
##### subset #####
# ids = ids[:50] # 
##################
if __name__=="__main__":
    all_feat, all_gt_ml = [], []
    for i in range(0, len(ids)):
        print('Loading File Number: '+str(i+1))
        data_cyc = np.load(path_main+add_cyc+ids[i], allow_pickle=True)
        data_time = np.load(path_main+add_time+ids[i], allow_pickle=True)
        
        ml_prep = ML_Preparing(pleth_cyc=data_cyc[0], abp_cyc=data_cyc[1], pleth=data_time[0], abp=data_time[1], idx_peak=data_cyc[3])
        
        dev1 = ml_prep.derivation()
        feat = ml_prep.extract_feat(dev1)
        
        ml_gt = ml_prep.extract_sbp_dbp(2, 110)
    
        all_feat.append(feat)
        all_gt_ml.append(ml_gt)
        
        '''
        np.save(target_path+"feature/"+ids[i], feat)
        np.save(target_path+"ground_truth_ml/"+ids[i], ml_gt)
        '''
    
    
    
    
    
    
    
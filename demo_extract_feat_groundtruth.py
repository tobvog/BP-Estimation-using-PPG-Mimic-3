import numpy as np
import os

from ML_Preparing import ML_Preparing

path_main = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/"
add_cyc = "cycled_data/"  
add_time = "preprocessed_data/"
files = os.listdir(path_main+add_time)  
target_path = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/feat/"

#all_dev, all_feat, all_gt_ml, all_gt_nn = [[] for i in range(0, 4)]
for i in range(0, len(files)):
    print('Loading File Number: '+str(i+1))
    data_cyc = np.load(path_main+add_cyc+files[i], allow_pickle=True)
    data_time = np.load(path_main+add_time+files[i], allow_pickle=True)
    
    ml_prep = ML_Preparing(data_cyc[0], data_cyc[1], data_time[1], data_cyc[3])
    
    dev1, dev2 = ml_prep.derivation()
    feat = ml_prep.extract_feat(dev1)
    
    ml_gt = ml_prep.extract_sbp_dbp(2, 110)
    nn_gt = ml_prep.extract_sbp_dbp(5, 110)
    '''
    all_dev.append([dev1, dev2])
    all_feat.append(feat)
    all_gt_ml.append(ml_gt)
    all_gt_nn.append(nn_gt)
    '''
    np.save(target_path+"derivations/dev1/"+files[i], dev1)
    np.save(target_path+"derivations/dev2/"+files[i], dev2)
    np.save(target_path+"feature/"+files[i], feat)
    np.save(target_path+"ground_truth_ml/"+files[i], ml_gt)
    np.save(target_path+"ground_truth_nn/"+files[i], nn_gt)

    
    
    
    
    
    
    
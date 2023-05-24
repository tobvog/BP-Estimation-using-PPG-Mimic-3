import matplotlib.pyplot as plt
import numpy as np
import os
from Elgendi_peak import ElgPeakDetection
from Preprocessing import Preprocessing_mimic3

path = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/"
arr = os.listdir(path+"preprocessed_data")
#path = "D:/MIMICIII_Database/preprocessed_data_slapnicar/"
#arr = os.listdir(path)
files = [x for x in arr if x.endswith('.npy')]

cycle_new, stats_all = [], []

nr_sub = len(files)

for i in range(0, nr_sub):
    print('Loading File Number: '+str(i+1))
    data = np.load(path+"preprocessed_data/"+files[i], allow_pickle=True)
    #data = np.load(path+files[i], allow_pickle=True)
    pleth = data[0]
    abp = data[1]

    print("Step 1: Peak Detection")
    peak_detect = ElgPeakDetection(data)
    idx_blocks, idx_peaks, pleth_new, abp_new = peak_detect.process()

    print("Step 2: Segment Cycles")
    preprocess = Preprocessing_mimic3([pleth_new, abp_new, data[2]], None)
    pleth_cycle, abp_cycle = preprocess.segment_cycles(idx_peaks, [30, 110])
   
    stats, flats = preprocess.detect_flat(data=[pleth_cycle, abp_cycle], edge_lines=0.1, edge_peaks=0.05)
    stats_all.append(stats)
    
    
    if stats[0]==False: # Ändern beim rechnen lassen
        idx_new = []
        for idx in range(0, len(pleth_cycle)):
            if flats[0][idx]==False and flats[1][idx]==False:
                idx_new.append(idx)
         
        pleth_cycle_new = pleth_cycle[idx_new]
        abp_cycle_new = abp_cycle[idx_new]
        idx_peaks_new = idx_peaks[idx_new]
    
    # np.save(path, [pleth_cycle_new, abp_cycle_new, data[3], idx_peaks_new])
    
            

#%%

idx_test = [0,5,6,8,10]
dummy = np.array([0,1,2,3,4,5,6,7,8,9,10])

res = dummy[idx_test]


res = 




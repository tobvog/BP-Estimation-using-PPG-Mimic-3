import numpy as np
import os
from Elgendi_peak import ElgPeakDetection
from Preprocessing import Preprocessing_mimic3

# Path of preprocessed data
path_data = "D:/MIMICIII_Database/preprocessed_data_slapnicar/"
# Target path
path_target = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/cycled_data/"
# Necessary subset of subject ids
ids = os.listdir(path_data)
### For subset ####
# ids = ids[:100] #
###################

stats_all = []
if __name__=="__main__":
    for i in range(0, len(ids)):
        
        # Loading data
        print('Loading File Number: ', i+1, " of ", len(ids))
        data = np.load(path_data+ids[i], allow_pickle=True)
        pleth = data[0]
        abp = data[1]
        
        # Detect peaks and correct data lengths
        print("Step 1: Peak Detection")
        peak_detect = ElgPeakDetection(data)
        idx_blocks, idx_peaks, pleth_new, abp_new = peak_detect.process()
    
        # Cycle Segmentation
        print("Step 2: Segment Cycles")
        preprocess = Preprocessing_mimic3([pleth_new, abp_new, data[2]], None)
        pleth_cycle, abp_cycle = preprocess.segment_cycles(idx_peaks, [30, 110])
        
        # Detect passing subjects
        stats, flats = preprocess.detect_flat(data=[pleth_cycle, abp_cycle], edge_lines=0.1, edge_peaks=0.05)
        stats_all.append(stats)
        
        # if subject passed, failed cycles need to filtered
        if stats[0]==True:
            print(ids[i], " passed")
            idx_new = []
            for idx in range(0, len(pleth_cycle)):
                if flats[0][idx]==False and flats[1][idx]==False:
                    idx_new.append(idx)
             
            # Save passed subject without failed cycles
            pleth_cycle_new = pleth_cycle[idx_new]
            abp_cycle_new = abp_cycle[idx_new]
            idx_peaks_new = idx_peaks[idx_new]
            # np.save(target_path+ids[i], [pleth_cycle_new, abp_cycle_new, data[2], idx_peaks_new])
        
        else:
            print(ids[i], " failed")
            
    # Saving statistical data        
    # np.save(target_path+"stats.npy", np.array(stats_all))
            





#%% Imports
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt, argrelextrema, welch
from scipy.stats import entropy, skew, kurtosis

import matplotlib.pyplot as plt
from Elgendi_peak import ElgPeakDetection

from Preprocessing import Preprocessing_mimic3

# functions
def design_filt(fs=125, lowcut=0.5, highcut=8, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output="sos") 
    
    return sos

#%%
#path = "D:/MIMICIII_Database/segmented_data_slapnicar/"
path = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/segmented_data/"
arr = os.listdir(path)    
files = [x for x in arr if x.endswith('.npy')]

nr_sub = len(files)
sos = design_filt()
for i in range(0,nr_sub):
    print('Loading File Number: '+str(i+1))
    data = np.load(path+files[i], allow_pickle=True)
    data_mod = [data[0,1], data[1,1], data[0,2]]
    
    preprocess = Preprocessing_mimic3(data, sos)
    print("Step 1/4: change_nan")
    preprocess.change_nan()
    #scaling
    print("Step 2/4: scaling")
    preprocess.scaling()
    # frequency filt
    print("Step 3/4: filtering(frequency)")
    preprocess.filt_freq()
    # hampelfilt
    print("Step 4/4: filtering(median)")
    preprocess.filt_median()
    
    pleth, abp, fs = preprocess.get_obj()

    path_extern = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/preprocessed_data"
    name = files[i]
    np.save(path_extern+name, [pleth, abp, fs])








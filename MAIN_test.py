#%% Imports
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt

import matplotlib.pyplot as plt
from Elgendi_peak import ElgPeakDetection

from Preprocessing import Preprocessing_mimic3
#%% function
def design_filt(lowcut=0.5, highcut=8, order=4):
    fs = 125
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output="sos") 
    
    return sos

def filt_freq(data, sos):
    return sosfiltfilt(sos, data)
    
#%% Main Test
path = "D:/MIMICIII_Database/segmented_data_slapnicar/"
#path = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/segmented_data/"
arr = os.listdir(path)    
files = [x for x in arr if x.endswith('.npy')]


X, Y = [], []
nr_sub = 15
sos = design_filt()
for i in range(1,nr_sub):
    print('Loading File Number: '+str(i+1))
    data = np.load(path+files[i], allow_pickle=True)
    
    abp = data[1,1]
    
    preprocess = Preprocessing_mimic3(data)
    print("Step 1/5: change_nan")
    pleth_nan_free = preprocess.change_nan()
    #scaling
    print("Step 2/5: scaling")
    pleth_scaled = preprocess.scaling(pleth_nan_free)
    #freqfilt
    #pleth_filt_freq = preprocess.freqfilt(pleth_scaled)
    print("Step 3/5: filtering(frequency)")
    pleth_filt_freq = filt_freq(pleth_scaled, sos)
    # hampelfilt
    print("Step 4/5: filtering(hampel)")
    pleth_filt_hamp = preprocess.filt_hampel(pleth_filt_freq)
    # create cycles
    print("Step 5/5: extract cycles via elgendi")
    peak_detect = ElgPeakDetection(pleth_filt_hamp, abp)

    data_square = peak_detect.squaring()
    ma_peak, l_w1 = peak_detect.moving_average(data_square, window = "w1")
    ma_beat, l_w2 = peak_detect.moving_average(data_square, window = "w2")

    pleth_mod, abp_mod, ma_peak_mod, data_square_mod = peak_detect.correct_length(pleth_filt_hamp, ma_peak, l_w1, l_w2)

    boi = peak_detect.boi(data_square_mod, ma_peak_mod, ma_beat)
    

    idx_blocks, idx_peaks = peak_detect.boi_onset_offset(boi, data_square_mod)
    
    pleth_cycle, abp_cycle = preprocess.segment_cycles(pleth_mod, abp_mod, idx_blocks)

    X.append(pleth_cycle)
    Y.append(abp_cycle)
    
    
    path_extern = "D:/MIMICIII_Database/preprocessed_data1_slapnicar/"
    name = "subject_"+str(i+1)
    np.save(path_extern+name, [X, Y])
    
    
#%%
plt.plot(X[0][25])
#%%
plt.plot(pleth_mod[11:114])

#%%
path_extern = "D:/MIMICIII_Database/preprocessed_data1_slapnicar/"
name = "subject_1"
np.save(path_extern+name, [X, Y])

#%%
from scipy.signal import argrelextrema



def find_nearest(array, value):
    a = list(array)
    return min(range(len(a)), key=lambda i: abs(a[i]- value))
    

min_idx = np.squeeze(np.array(argrelextrema(pleth_scaled, np.less)))

idx_temp = find_nearest(min_idx, 500)











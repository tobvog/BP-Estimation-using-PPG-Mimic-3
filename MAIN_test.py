#%% Imports
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt, argrelextrema, welch
from scipy.stats import entropy, skew, kurtosis

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
    
def find_dianotch(data, data_diff):
    try:
        max_ = np.argwhere(data == max(data))
        if min(data_diff[max_[0,0]+2:int(len(data)*0.75)]) < 0.005:
            result = np.squeeze(np.argwhere(data_diff == min(data_diff[max_[0,0]+2:len(data)-20])))
        else:
            result = 0
    except:
        result = 0
        
    return result

def spec_flatten(X,Y):
    X_flat = []
    Y_flat = []
    for i in range(0, len(X)):
        print(i)
        for j in range(0, len(X[i])):
            print(j)
            x_temp = X[i][j]
            y_temp = Y_mod[i][j]
            print(y_temp)
            if np.isnan(y_temp[0]) == False and np.isnan(y_temp[1]) == False:
                X_flat.append(x_temp)
                Y_flat.append(y_temp)
    
    X_flat = np.array(X_flat)
    Y_flat = np.array(Y_flat)
     
    return X_flat, Y_flat

def detect_flat(pleth, abp, edge_lines=0.1, edge_peaks=0.05):
    flat_lines, flat_peaks, max_list = [], [], []
    
    for cycle in range(0, len(abp)):
        temp_lines, temp_peaks = [], []
        data_max = argrelextrema(abp[cycle], np.greater)
        max_list.append(data_max)

        for sample in range(0, len(abp[cycle])-2):
            if len(data_max[0]) == 0:
                temp_peaks.append(False)
                temp_lines.append(True)
                break
            else:
                if sample in data_max[0]:   
                    #temp_lines.append(False)
                    if sample == len(abp[cycle])-2:
                        if abp[cycle][sample] == abp[cycle][sample+1] and abp[cycle][sample] == abp[cycle][sample+2]:# and abp[cycle][sample] == abp[cycle][sample-1] :
                            temp_peaks.append(True)
                            temp_lines.append(True)
                            break          
        
                    if sample < len(abp[cycle])-2:
                        if abp[cycle][sample] == abp[cycle][sample+1] and abp[cycle][sample] == abp[cycle][sample+2]:# and abp[cycle][sample] == abp[cycle][sample+3]:
                            temp_peaks.append(True)
                            temp_lines.append(True)
                            break
        
                else:
                    #temp_peaks.append(False)
                    if sample == len(abp[cycle])-2:
                        if abp[cycle][sample] == abp[cycle][sample+1] and abp[cycle][sample] == abp[cycle][sample+2]:# and abp[cycle][sample] == abp[cycle][sample-1]:
                            temp_lines.append(True)
                            break
        
           
                    if sample < len(abp[cycle])-2:
                        if abp[cycle][sample] == abp[cycle][sample+1] and abp[cycle][sample] == abp[cycle][sample+2]:# and abp[cycle][sample] == abp[cycle][sample+3]:
                            temp_lines.append(True)
                            break

        flat_lines.append(temp_lines)
        flat_peaks.append(temp_peaks)
      
    fquote_lines = flat_lines.count([True])/len(flat_lines)
    fquote_peaks = flat_peaks.count([True])/len(flat_peaks)

    if fquote_lines > edge_lines or fquote_peaks > edge_peaks:
        passed = False
    else:
        passed = True
            
    return [passed, fquote_lines, fquote_peaks]
#%% Main Test
path = "D:/MIMICIII_Database/segmented_data_slapnicar/"
#path = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/segmented_data/"
arr = os.listdir(path)    
files = [x for x in arr if x.endswith('.npy')]


x, y, all_stats = [], [], []
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

    stats = preprocess.detect_flat(pleth_cycle, abp_cycle)

    all_stats.append(stats)
    x.append(pleth_cycle)
    y.append(abp_cycle)
    
    
    path_extern = "D:/MIMICIII_Database/preprocessed_data1_slapnicar/"
    name = "subject_"+str(i+1)
    np.save(path_extern+name, [x, y])
    
np.save("D:/MIMICIII_Database/detect_flats_stats") # Change path
#%%
path_extern = "D:/MIMICIII_Database/preprocessed_data1_slapnicar/"
name = "subject_1"
np.save(path_extern+name, [X, Y])

#%% Load Data

path = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/cycled_data/"
arr = os.listdir(path) 
files = [x for x in arr if x.endswith('.npy')]

X, Y = [], []
for i in range(0, len(files)):
    temp = np.load(path+files[i], allow_pickle=True)
    
    X.append(temp[0])
    Y.append(temp[1])
#%% Differing the data

X_diff1,  X_diff2 = [], []
for sub in range(0, len(X)):
    temp_sub1, temp_sub2 = [], []
    for cyc in range(0, len(X[sub])):
        cycle = X[sub][cyc]
        cycle_diff1 = np.diff(cycle)
        cycle_diff2 = np.diff(cycle_diff1)
        
        temp_sub1.append(np.array(cycle_diff1))
        temp_sub2.append(np.array(cycle_diff2))
        
    X_diff1.append(temp_sub1)
    X_diff2.append(temp_sub2)
#%%
path = "D:/MIMICIII_Database/preprocessed_data1_slapnicar/"
#path = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/segmented_data/"
arr = os.listdir(path)    
files = [x for x in arr if x.endswith('.npy')]
stats = []
for sub in files:
    print(sub)
    data = np.load(path+sub, allow_pickle=True)
    pleth = data[0]
    abp = data[1]
    
    passed, qline, qpeak = detect_flat(pleth, abp)
    stats.append([sub, passed, qline, qpeak])
    







#%%
fig, axs = plt.subplots(3)
#fig.suptitle('Vertically stacked subplots')

sub = 0
cyc = 3503
axs[0].plot(X[sub][cyc])
axs[1].plot(X_diff1[sub][cyc])
axs[2].plot(X_diff2[sub][cyc])

#%%
sub = 0
cyc = 55

plt.plot(Y[sub][cyc])


#%% Create Y

Y_mod = []
for sub in range(0, len(Y)):
    #print(sub)
    subject = []
    for cyc in range(0, len(Y[sub])):
        #print(cyc)
        temp = Y[sub][cyc]
        sbp = max(temp)        
        '''
        idx_sbp = np.squeeze(np.where(temp == sbp))
        try:
            dbp = min(temp[idx_sbp:])
        except:
            dbp = min(temp[idx_sbp[0]:])
        '''    
        dbp = min(temp)
        
        subject.append([sbp, dbp])
    Y_mod.append(subject)

Y_mod = np.array(Y_mod, dtype=object) 

#%%
X_feat = []
for sub in range(0, len(X)):
    print("subject: "+str(sub))
    temp_feat = []
    for cyc in range(0, len(X[sub])):
        #print("Cycle: "+str(cyc))
        cycle = X[sub][cyc]
        cycle_diff1 = X_diff1[sub][cyc]
        cycle_diff2 = X_diff2[sub][cyc]
        ## Time Features       
        # Cycle duration time 
        t_c = len(cycle)
        # Time from cycle start to cycle end
        t_s = np.squeeze(np.array(np.where(cycle == max(cycle))))
        #t_s = cycle.index(max(cycle))
        # Time from systolic peak to cycle end
        t_d = t_c-t_s
        # Time from cycle start to first peak in PPG’ (steepest point)
        diff1_peaks = argrelextrema(cycle_diff1, np.greater)
        if len(diff1_peaks[0]) == 0:
            t_steepest = 0
        else:    
            t_steepest = diff1_peaks[0][0]
        # Time from cycle start to second peak in PPG’ (dicrotic notch)       
        t_dianotch = find_dianotch(cycle, cycle_diff1)
        # if t_dianotch != 0 and len(diff1_peaks)>1:
        #     if t_dianotch == diff1_peaks[1]:
        #         print("Alright") 
        #     else:
        #         print("all wrong")
            
        # Time from systolic peak to dicrotic notch
        if t_dianotch != 0:
            t_systodianotch = t_dianotch - t_s
        else:
            t_systodianotch = 0
        # Time from dicrotic notch to cycle end
        if t_dianotch != 0:
            t_diatoend = t_c - t_dianotch 
        else:
            t_diatoend = 0
        # Ratio between systolic and diastolic amplitude
        if t_dianotch != 0:          
            dia_peak = argrelextrema(cycle_diff1[t_dianotch:], np.greater)
            if np.shape(dia_peak) != (1,0):
                ratio = cycle[t_s]/cycle[dia_peak[0][0]]
            else:
                ratio = 0
        else:
            ratio = 0
        
        ## Frequency Features
        # Three peaks with the largest magnitude from the PSD were
        # considered. These tell us the dominant frequencies in the cycle. Both the magnitude
        # values and the frequencies (in Hz) were taken as features.
        freq, psd = welch(cycle, 125, nperseg=len(cycle))

        sorted_data = np.sort(psd)
        psd1 = sorted_data[-1]
        psd2 = sorted_data[-2]
        psd3 = sorted_data[-3]

        freq1 = np.squeeze(freq[np.where(psd == psd1)])
        freq2 = np.squeeze(freq[np.where(psd == psd2)])
        freq3 = np.squeeze(freq[np.where(psd == psd3)])
        # Calculated as the sum of the squared fast Fourier transform (FFT) component
        # magnitudes. The energy was then normalized by dividing it with the cycle length.
        freq_fft = np.abs((np.real(np.fft.fft(cycle))))
        data_used = np.square(freq_fft[:int(len(freq_fft)*0.5)])
        
        energy = np.sum(data_used)/len(data_used)
        # Entropy
        data_used = freq_fft[:int(len(freq_fft)*0.5)]
        data_used_norm = []
        for i in range(0, len(data_used)):
            temp = data_used[i]/max(data_used)
            data_used_norm.append(temp)
            
        data_used_norm = np.array(data_used_norm)

        entropy_ = entropy(data_used_norm)
        # A normalized histogram, which is essentially the distribution of the FFT magnitudes into 10 equal sized bins ranging from 0 Hz to 62.5 Hz. 
        data_used = freq_fft[:63]
        hist = np.histogram(data_used, 10)
        binneddistribution = hist[0]/10
        bin0 = binneddistribution[0]
        bin1 = binneddistribution[1]
        bin2 = binneddistribution[2]
        bin3 = binneddistribution[3]
        bin4 = binneddistribution[4]
        bin5 = binneddistribution[5]
        bin6 = binneddistribution[6]
        bin7 = binneddistribution[7]
        bin8 = binneddistribution[8]
        bin9 = binneddistribution[9]
        # Skewness and kurtosis. These describe the shape of the cycle. More precisely, skewness tells
        # us about the symmetry while kurtosis tells us about the flatness.
        data_used = freq_fft[:int(len(freq_fft)*0.5)]
        skewness = skew(data_used)
        kurtosis_ = kurtosis(data_used)
        
        feat = np.array([t_c, t_s, t_d, t_steepest, t_dianotch, t_systodianotch, t_diatoend, ratio, 
                         psd1, psd2, psd3, freq1, freq2, freq3, energy, entropy_,
                         bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9, 
                         skewness, kurtosis_], 
                        dtype=object)
        temp_feat.append(feat)
    X_feat.append(temp_feat)
 
X = np.array(X_feat, dtype=object)


Xf, Yf = spec_flatten(X, Y_mod)
#%%

X_flat = []
Y_flat = []
for i in range(0, len(X)):
    print(i)
    for j in range(0, len(X[i])):
        print(j)
        x_temp = X[i][j]
        y_temp = Y_mod[i][j]
        print(y_temp)
        if np.isnan(y_temp[0]) == False and np.isnan(y_temp[1]):
            X_flat.append(x_temp)
            Y_flat.append(y_temp)

X_flat = np.array(X_flat)
Y_flat = np.array(Y_flat)

#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

clf = RandomForestRegressor()
loo = LeaveOneOut()
loo.get_n_splits(X)

Yf_sbp = Yf[:,0]
Yf_dbp = Yf[:,1]

all_mse = []
for train_index, test_index in loo.split(Xf):
     print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = Xf[train_index], Xf[test_index]
     Y_train, Y_test = Yf_sbp[train_index], Yf_sbp[test_index]
     
     clf.fit(X_train, Y_train)
     y_pred = clf.predict(X_test)
     
     mse = mean_squared_error(Y_test, y_pred)
     
     all_mse.append(mse)
    

        
#%%
    
    
    #%%
plt.plot(X_train[40500])








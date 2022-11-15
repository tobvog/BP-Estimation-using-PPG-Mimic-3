'''
Class for machine learning

'''
#%% Imports
import numpy as np

from scipy.signal import argrelextrema, welch
from scipy.stats import entropy, skew, kurtosis
#%%

class ML_Preparing:
    
    def __init__(self, pleth, abp, fs):
        self.pleth = pleth
        self.abp = abp
        self.fs = fs
###############################################################################
###############################################################################
###############################################################################        
    def derivation(self):
        dev1, dev2 = [], []
        
        for sub in range(0, len(self.pleth)):
            temp_sub1, temp_sub2 = [], []
            for cyc in range(0, len(self.pleth[sub])):
                cycle = self.pleth[sub][cyc]
                cycle_dev1 = np.diff(cycle)
                cycle_dev2 = np.diff(cycle_dev1)
                
                temp_sub1.append(np.array(cycle_dev1))
                temp_sub2.append(np.array(cycle_dev2))
                
            dev1.append(temp_sub1)
            dev2.append(temp_sub2)
            
        return np.array(dev1, dtype=object), np.array(dev2, dtype=object)
###############################################################################
###############################################################################
###############################################################################
    def extract_feat(self, dev1):      
        def find_dianotch(data, data_diff):
            try:
                max_ = np.argwhere(data == max(data))
                if min(data_diff[max_[0,0]+2:int(len(data)*0.75)]) < 0.005:
                    result = np.squeeze(np.argwhere(data_diff == min(data_diff[max_[0,0]+2:len(data)-20])))
                else:           # TEST IF ITS NECESSARY
                    result = 0  #        !!!!    
            except:
                result = 0               
            return result
        
        result = []
        for sub in range(0, len(self.pleth)):
            print("Feature extraction of subject: "+str(sub+1)+" of "+ str(len(dev1)))
            temp_sub = []
            for cyc in range(0, len(self.pleth[sub])):
                #print("Cycle: "+str(cyc))
                cycle = self.pleth[sub][cyc]
                cycle_dev1 = dev1[sub][cyc]
                
                ''' 
                Time Features
                '''
                # Cycle duration time 
                t_c = len(cycle)
                
                # Time from cycle start to systic peak
                t_s = np.squeeze(np.array(np.where(cycle == max(cycle))))
                
                # Time from systolic peak to cycle end
                t_d = t_c-t_s
                
                # Time from cycle start to first peak in PPG’ (steepest point)
                dev1_peaks = argrelextrema(cycle_dev1, np.greater)
                if len(dev1_peaks[0]) == 0:
                    t_steepest = 0
                else:    
                    t_steepest = dev1_peaks[0][0]
                # Time from cycle start to second peak in PPG’ (dicrotic notch)       
                t_dianotch = find_dianotch(cycle, cycle_dev1)
                    
                # Time from systolic peak to dicrotic notch
                if t_dianotch != 0:
                    # Time from systolic peak to dicrotic notch
                    t_systodianotch = t_dianotch - t_s
                    
                    # Time from dicrotic notch to cycle end
                    t_diatoend = t_c - t_dianotch 
                    
                    # Ratio between systolic and diastolic amplitude
                    dia_peak = argrelextrema(cycle_dev1[t_dianotch:], np.greater)
                    if np.shape(dia_peak) != (1,0):
                        ratio = cycle[t_s]/cycle[dia_peak[0][0]]
                    else:
                        ratio = 0                   
                else:
                    t_systodianotch = 0
                    t_diatoend = 0
                    ratio = 0
                    
                '''
                Frequency Features
                '''
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
                                 skewness, kurtosis_], dtype=object)
                #temp_sub = np.append(temp_sub, feat)
                temp_sub.append(feat)
            result.append(temp_sub)
         
        return np.array(result, dtype=object)
###############################################################################
###############################################################################
###############################################################################    
    def extract_sbp_dbp(self):
        result = []
        for sub in range(0, len(self.abp)):
            #print(sub)
            subject = []
            for cyc in range(0, len(self.abp[sub])):
                #print(cyc)
                temp = self.abp[sub][cyc]
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
            result.append(subject)

        return np.array(result, dtype=object)
 ###############################################################################
 ###############################################################################
 ###############################################################################     

        
        
        
        
        
        
        
        


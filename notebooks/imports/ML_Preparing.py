import numpy as np

from scipy.signal import argrelextrema, welch
from scipy.stats import entropy, skew, kurtosis

## @brief Class for the preparing of the data for machine learning.  
## @details This class provides methods to prepare data for machine learning by Slapnicar et al.
class ML_Preparing:   
    def __init__(self, pleth_cyc=None, abp_cyc=None, pleth=None, abp=None, idx_peak=None, fs=125):
        ##
        # @brief This constructor initalizes the ML_Preparing object.
        # @param pleth_cyc      PPG input data as cycle. Default=None.
        # @param abp_cyc        Blood pressure input data as cycle. Default=None.              
        # @param pleth          PPG input data as one time line per subject. Default=None. 
        # @param abp            Blood pressure input data as one time line per subject. Default=None.
        # @param idx_peak       Array with indices of peaks. The Peaks represent the systolic blood pressure. Default=None.
        # @param fs             Sampling rate of input data. Default=125 
        ##
        self.pleth_cyc = pleth_cyc
        self.abp_cyc = abp_cyc
        self.pleth = pleth
        self.abp = abp
        self.idx_peak = idx_peak
        self.fs = fs
###############################################################################
###############################################################################
###############################################################################        
    def derivation2(self):
        ##
        # @brief    This method calculate the first and second derivation of one subject.            
        # @return   The first and second derivation of all cycle of one subject
        ##
        dev1, dev2 = [], []
        
        for cycle in self.pleth_cyc:
            cyc_dev1 = np.diff(cycle)
            cyc_dev1 = np.append(cyc_dev1, cyc_dev1[-1])
            cyc_dev2 = np.diff(cyc_dev1)
            cyc_dev2 = np.append(cyc_dev2, cyc_dev2[-1])
            dev2.append(cyc_dev2)
            dev1.append(cyc_dev1)
            
        return np.array(dev1, dtype=object), np.array(dev2, dtype=object) 

###############################################################################
###############################################################################
###############################################################################        
    def derivation(self):
        ##
        # @brief    This method calculate the first derivation of one subject.            
        # @return   The first derivation of all cycle of one subject
        ##
        dev1 = []
        
        for cycle in self.pleth_cyc:
            cyc_dev1 = np.diff(cycle)
            dev1.append(cyc_dev1)
            
        return np.array(dev1, dtype=object)
    
###############################################################################
###############################################################################
###############################################################################
    @staticmethod
    def __find_dianotch(data, data_diff):
        try:
            max_ = np.argwhere(data == max(data))
            if min(data_diff[max_[0,0]+2:int(len(data)*0.75)]) < 0.005:
                result = np.squeeze(np.argwhere(data_diff == min(data_diff[max_[0,0]+2:len(data)-20])))
            else:           
                result = 0   
        except:
            result = 0               
        return result
    
###############################################################################
###############################################################################
###############################################################################
    def extract_feat(self, dev1): 
        ##
        # @brief        This method calculate all necessary feature for the classical machine learning by Slapnicar et al.
        # @param dev1   The first derivation of all cycles of one subject.          
        # @return       Feature array.
        ##
        
        result = []
        for cyc in range(0, len(self.pleth_cyc)):
            #print("Cycle: "+str(cyc))
            cycle = self.pleth_cyc[cyc]
            cycle_dev1 = dev1[cyc]
            
            ''' 
            Time Features
            '''
            # Cycle duration time 
            t_c = len(cycle)
            
            # Time from cycle start to systic peak
            t_s = np.squeeze(np.array(np.where(cycle == max(cycle))))
            if type(t_s)!=int:
                t_s = t_s[0]
            
            # Time from systolic peak to cycle end
            t_d = t_c-t_s
            
            # Time from cycle start to first peak in PPG’ (steepest point)
            dev1_peaks = argrelextrema(cycle_dev1, np.greater)
            if len(dev1_peaks[0]) == 0:
                t_steepest = 0
            else:    
                t_steepest = dev1_peaks[0][0]
            # Time from cycle start to second peak in PPG’ (dicrotic notch)       
            t_dianotch = self.__find_dianotch(cycle, cycle_dev1)
                
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
            
            feat = np.array([t_c, t_s, t_d, t_steepest, int(t_dianotch), t_systodianotch, t_diatoend, ratio, 
                             psd1, psd2, psd3, int(freq1), int(freq2), int(freq3), energy, entropy_,
                             bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9, 
                             skewness, kurtosis_], dtype=object)
            
            result.append(feat)
         
        return np.array(result, dtype=object)
###############################################################################
###############################################################################
###############################################################################

    def extract_sbp_dbp(self, window, pad, epoch=False):
        ##
        # @brief            This method extract the systolic and diastolic blood pressure of the blood pressure data. Additionally its able to extract cycles for the resnet by slapnicar et al.
        # @param window     Used window size in seconds.    
        # @pad              Number of sample after the systolic blood pressure which should be used to search for diastolic blood pressure.
        # @return           Array of ground truth data for blood pressure and array of cycles.
        ##
        data = self.abp
        result = []
        distance = int((window*self.fs)/2)
        pleth_cyc = [] 
        
        for i in range(0, len(self.idx_peak)):
            main_p = self.idx_peak[i] 
            idx_min = main_p-distance
            idx_max = main_p+distance
                        
            if len(self.pleth[idx_min:idx_max]) != 624 and epoch==True:
                continue
            
            status = "forward"
            peaks_max = [main_p]
            ct = 1
            while(status=="forward"):
                try:
                    if self.idx_peak[i+ct]<=idx_max:
                        peaks_max.append(self.idx_peak[i+ct])
                        ct+=1
                    elif self.idx_peak[i+ct]>idx_max:
                        status="backward"
                except:
                    status="backward"
                    
            ct = 1
            while(status=="backward"):
                try:
                    if self.idx_peak[i-ct]>=idx_min:
                        peaks_max.append(self.idx_peak[i-ct])
                        ct+=1
                    elif self.idx_peak[i-ct]<idx_min:
                        status="finish"
                except:
                    status="finish"
        
            peaks_min = []
            for i in peaks_max:
                seq = data[i:i+pad]
                peaks_min.append(min(seq))
                
            
            sbp = np.mean(data[peaks_max])
            dbp = np.mean(peaks_min)
            
            result.append([sbp, dbp])
            
            if epoch==True:
                pleth_cyc.append(self.pleth[idx_min:idx_max])

                       
        if epoch==True:
            return np.array(result), np.array(pleth_cyc)
        else:          
            return np.array(result)
                          
        
        
        
        
        
        
        


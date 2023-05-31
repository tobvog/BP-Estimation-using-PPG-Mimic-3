import numpy as np
import pandas as pd

from hampel import hampel
from scipy.signal import butter, sosfiltfilt, medfilt

class Preprocessing_mimic3:
    def __init__(self, data, sos):
        self.pleth = data[0]
        self.abp =data[1]
        self.fs = data[2]
        self.sos = sos
 
###############################################################################
###############################################################################
###############################################################################
    def get_obj(self):
        return self.pleth, self.abp, self.fs
    
###############################################################################
###############################################################################
###############################################################################
    def design_filt(self, lowcut=0.5, highcut=8, order=4):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='band', output="sos") 
        return sos
    
###############################################################################
###############################################################################
###############################################################################
    def filt_freq(cls):
        cls.pleth = sosfiltfilt(cls.sos, cls.pleth)
        
###############################################################################
###############################################################################
###############################################################################
    def scaling(cls):
        mean = np.nanmean(cls.pleth)
        std = np.nanstd(cls.pleth)
        cls.pleth = (cls.pleth-mean)/std
       
###############################################################################
###############################################################################
###############################################################################  
    def change_nan(cls):
        data = cls.pleth
        result = data
        
        for i in range(0,len(data)):
            if np.isnan(data[i]) == True and np.isnan(data[i+1]) == False:
                result[i] = int((result[i-1]+result[i+1])/2)
                
            elif np.isnan(data[i]) == True and np.isnan(data[i+1]) == True and np.isnan(data[i+2]) == False:
                result[i] = int((result[i-1]+result[i+2])/2)
                result[i+1] = int((result[i-1]+result[i+2])/2)
                
            elif np.isnan(data[i]) == True and np.isnan(data[i+1]) == True and np.isnan(data[i+2]) == True:
                result[i] = 0
        cls.pleth = data
        
###############################################################################
###############################################################################
###############################################################################
    def filt_hampel(self, data, hampel_window=3):
        df = pd.Series(data)
        result  = hampel(df, hampel_window, n=3, imputation=True)
        
        return result
    
###############################################################################
###############################################################################
###############################################################################
    def filt_median(cls, k_size=3):
        cls.pleth = medfilt(cls.pleth, k_size)    
        
###############################################################################
###############################################################################
###############################################################################     
    def detect_flat(self, data, edge_lines=0.1, edge_peaks=0.05):
        flat_lines, flat_peaks = [], []
        pleth = data[0]
        abp = data[1]
        
        for cycle in range(0, len(abp)):
            abp_max = max(abp[cycle])
            ct_max = np.count_nonzero(abp[cycle]==abp_max)
                                  
            for sample in range(0, len(abp[cycle])-2):
                if ct_max > 2:
                    flat_peaks.append(True)
                    flat_lines.append(True)
                    break
                       
                if sample < len(abp[cycle])-3:
                    if abp[cycle][sample]==abp[cycle][sample+1] and abp[cycle][sample]==abp[cycle][sample+2] or (pleth[cycle][sample]==pleth[cycle][sample+1] and pleth[cycle][sample]==pleth[cycle][sample+2]):# and abp[cycle][sample] == abp[cycle][sample+3]:
                        flat_peaks.append(False)
                        flat_lines.append(True)
                        break
                     
                else:
                    if abp[cycle][sample]==abp[cycle][sample+1] and abp[cycle][sample]==abp[cycle][sample+2] or (pleth[cycle][sample]==pleth[cycle][sample+1] and pleth[cycle][sample]==pleth[cycle][sample+2]):# and abp[cycle][sample] == abp[cycle][sample-1]:
                        flat_peaks.append(False)
                        flat_lines.append(True)
                        break
                    else:
                        flat_peaks.append(False)
                        flat_lines.append(False)
                        break
        if len(flat_lines)==0 or len(flat_peaks)==0:
            fquote_lines = 1
            fquote_peaks = 1
        else:
            fquote_lines = flat_lines.count(True)/len(flat_lines)
            fquote_peaks = flat_peaks.count(True)/len(flat_peaks)
            
          
        fquote_lines = flat_lines.count(True)/len(flat_lines)
        fquote_peaks = flat_peaks.count(True)/len(flat_peaks)
        print("Lines: "+str(fquote_lines))
        print("Peaks: "+str(fquote_peaks))
    
        if fquote_lines > edge_lines or fquote_peaks > edge_peaks:
            passed = False
        else:
            passed = True
                
        return [passed, fquote_lines, fquote_peaks], [flat_lines, flat_peaks]
    
###############################################################################
###############################################################################
###############################################################################      
    def segment_cycles(self, peak_idx, pad):
        
        cycle_pleth, cycle_abp = [],[]
        for i in range(0, len(peak_idx)):
            temp_cyc1 = self.pleth[(peak_idx[i]-pad[0]):peak_idx[i]]
            temp_cyc2 = self.pleth[peak_idx[i]:(peak_idx[i]+pad[1])] 
            try:
                min1 = np.where(temp_cyc1==temp_cyc1.min())
                min2 = np.where(temp_cyc2==temp_cyc2.min())
                idx1 = min1[0][0]+peak_idx[i]-pad[0]
                idx2 = min2[0][-1]+peak_idx[i]
                cycle_pleth.append(self.pleth[idx1:idx2])
                cycle_abp.append(self.abp[idx1:idx2])
            except:
                continue
            
        return np.array(cycle_pleth, dtype=object), np.array(cycle_abp, dtype=object)

        
        
        
    
    
    
    
    
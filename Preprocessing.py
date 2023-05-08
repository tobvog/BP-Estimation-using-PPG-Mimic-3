import numpy as np
import pandas as pd

from hampel import hampel
from scipy.signal import argrelextrema, butter, sosfiltfilt, medfilt

class Preprocessing_mimic3:
    def __init__(self, data, sos):
        self.pleth = data[0]
        self.abp =data[1]
        self.fs = data[2]
        self.sos = sos
        
        '''
        Muss verallgemeinert werden
        
        self.pleth = data[0,1]
        self.abp = data[1,1]
        self.fs = data[0,2]   
        self.sos = sos
        '''
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
        
        for cycle in range(0, len(data)):
            temp_lines, temp_peaks = [], []
            data_max = argrelextrema(data[cycle], np.greater)
            #pleth_max = argrelextrema(self.abp[cycle], np.greater)
            #max_list.append(data_max)
    
            for sample in range(0, len(data[cycle])-2):
                if len(data_max[0]) == 0:
                    temp_peaks.append(False)
                    temp_lines.append(True)
                    break
                else:
                    if sample in data_max[0]:   
                        #temp_lines.append(False)
                        if sample == len(data[cycle])-2:
                            if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2]:# and abp[cycle][sample] == abp[cycle][sample-1] :
                                temp_peaks.append(True)
                                temp_lines.append(True)
                                break          
            
                        if sample < len(data[cycle])-2:
                            if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2]:# and abp[cycle][sample] == abp[cycle][sample+3]:
                                temp_peaks.append(True)
                                temp_lines.append(True)
                                break
            
                    else:
                        #temp_peaks.append(False)
                        if sample == len(data[cycle])-2:
                            if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2]:# and abp[cycle][sample] == abp[cycle][sample-1]:
                                temp_lines.append(True)
                                break
            
               
                        if sample < len(data[cycle])-2:
                            if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2]:# and abp[cycle][sample] == abp[cycle][sample+3]:
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
###############################################################################
###############################################################################
###############################################################################      
    def segment_cycles(self, peak_idx, pad):
        
        cycle_pleth, cycle_abp = [],[]
        for i in range(0, len(peak_idx), 2):
            temp_cyc1 = self.pleth[peak_idx[i]-pad[0]:peak_idx[i]]
            temp_cyc2 = self.pleth[peak_idx[i]:peak_idx[i]+pad[1]] 
            try:
                min1 = np.where(temp_cyc1==temp_cyc1.min())
                min2 = np.where(temp_cyc2==temp_cyc2.min())
                #print(min1[0][0])
                #print(min2[0][0])
                cycle_pleth.append(self.pleth[min1[0][0]:min2[0][-1]])
                cycle_abp.append(self.abp[min1[0][0]:min2[0][-1]])
            except:
                continue
            
        return np.array(cycle_pleth, dtype=object), np.array(cycle_abp, dtype=object)
###############################################################################
###############################################################################
###############################################################################        
    def process(self):

        pleth_nan_free = self.change_nan()
        
        pleth_scaled = self.scaling(pleth_nan_free)
        
        pleth_filt_freq = self.freqfilt(pleth_scaled)
        
        pleth_filt_hamp = self.filt_hampel(pleth_filt_freq)
                
        pleth_cycle, abp_cycle = self.segment_cycles(pleth_filt_hamp)
        
        x, y = self.detect_flat(pleth_cycle, abp_cycle)
        
        return x, y 
        
        
        
    
    
    
    
    
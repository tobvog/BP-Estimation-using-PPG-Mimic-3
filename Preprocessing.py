import numpy as np
import pandas as pd

from scipy import signal
from hampel import hampel
from scipy.signal import argrelextrema

from Elgendi_peak import ElgPeakDetection

class Preprocessing_mimic3:
    def __init__(self, data):
        self.pleth = data[:,0,1]
        self.abp = data[:,1,1]
        self.fs = data[:,0,2]
###############################################################################
###############################################################################
###############################################################################
    def freqfilt(self, data, lowcut=0.5, highcut=8, order=4):
        
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high], btype='band', output="sos") 

        result = []        
        for sub in range(0,len(data)):
            temp = signal.sosfiltfilt(sos, data[sub])
            result.append(temp)
        result = np.array(result, dtype=object)
            
        return result
###############################################################################
###############################################################################
###############################################################################
    def scaling(data):
        result = []
        for sub in range(0,len(data)):
            mean = np.nanmean(data[sub])
            var = np.nanvar(data[sub])
            temp = (data[sub]-mean)/var
            result.append(temp)  
        result = np.array(result, dtype=(object))
        
        return result
###############################################################################
###############################################################################
###############################################################################
    def change_nan(self):
        result_all = []
        data = self.pleth
        for sub in range(0,len(data)):
            print("Subject "+str(sub+1))
            temp = data[sub]
            result = temp
            
            for i in range(0,len(temp)):
                if np.isnan(temp[i]) == True and np.isnan(temp[i+1]) == False:
                    result[i] = int((result[i-1]+result[i+1])/2)
                    
                elif np.isnan(temp[i]) == True and np.isnan(temp[i+1]) == True and np.isnan(temp[i+2]) == False:
                    result[i] = int((result[i-1]+result[i+2])/2)
                    result[i+1] = int((result[i-1]+result[i+2])/2)
                    
                elif np.isnan(temp[i]) == True and np.isnan(temp[i+1]) == True and np.isnan(temp[i+2]) == True:
                    result[i] = 0
                    
            result_all.append(result)
        result_all = np.array(result_all, dtype=object)
        
        return result_all
###############################################################################
###############################################################################
###############################################################################
    def filt_hampel(self, data):
        hampel_window = 3
        
        hampel_filt_data = []
        for sub in range(0,len(data)):
            df = pd.Series(data[sub])
            print("Subject No. "+str(sub+1))
            temp  = hampel(df, hampel_window, n=3, imputation=True)
            hampel_filt_data.append(temp)
            
        result = np.array(hampel_filt_data, dtype=object)
        
        return result
###############################################################################
###############################################################################
############################################################################### 
    # def detect_flat(self, abp):
    #     abp_flat_lines,  number_of_bvalue_line, abp_flat_peaks, number_of_bvalue_peak = [], [], [], []

    #     for sub in range(0,len(abp)):
    #         print("Subject number: "+str(sub+1))
    #         data = abp[sub]
    #         temp_line, temp_peak = [], []
    #         data_max = argrelextrema(data, np.greater)
    #         for sample in range(0, len(data)):
    #             if sample in data_max[0]:   
    #                 temp_line.append(False)
    #                 if sample == len(data)-3:
    #                     if data[sample] == data[sample+1] and data[sample] == data[sample+2] and data[sample] == data[sample-1] :
    #                         temp_peak.append(True)
    #                         temp_line.append(True)
    #                     else:
    #                         temp_peak.append(False)
    #                         temp_line.append(False)
    #                 elif sample == len(data)-2:
    #                     if data[sample] == data[sample+1] and data[sample] == data[sample-1] and data[sample] == data[sample-2]:
    #                         temp_peak.append(True)
    #                         temp_line.append(True)
    #                     else:
    #                         temp_peak.append(False)
    #                         temp_line.append(False)
    #                 elif sample == len(data)-1:
    #                     if data[sample] == data[sample-1] and data[sample] == data[sample-2] and data[sample] == data[sample-3]:
    #                         temp_peak.append(True)
    #                         temp_line.append(True)
    #                     else:
    #                         temp_peak.append(False)  
    #                         temp_line.append(False)
    #                 else:
    #                     if data[sample] == data[sample+1] and data[sample] == data[sample+2] and data[sample] == data[sample+3]:
    #                         temp_peak.append(True)
    #                         temp_line.append(True)
    #                     else:
    #                         temp_peak.append(False)
    #                         temp_line.append(False)
    #             else:
    #                 temp_peak.append(False)
    #                 if sample == len(data)-3:
    #                     if data[sample] == data[sample+1] and data[sample] == data[sample+2] and data[sample] == data[sample-1]:
    #                         temp_line.append(True)
    #                     else:
    #                         temp_line.append(False)
    #                 elif sample == len(data)-2:
    #                     if data[sample] == data[sample+1] and data[sample] == data[sample-1] and data[sample] == data[sample-2]:
    #                         temp_line.append(True)
    #                     else:
    #                         temp_line.append(False)  
    #                 elif sample == len(data)-1:
    #                     if data[sample] == data[sample-1] and data[sample] == data[sample-2] and data[sample] == data[sample-3]:
    #                         temp_line.append(True)
    #                     else:
    #                         temp_line.append(False)        
    #                 else:
    #                     if data[sample] == data[sample+1] and data[sample] == data[sample+2] and data[sample] == data[sample+3]:
    #                         temp_line.append(True)
    #                     else:
    #                         temp_line.append(False)
              
    #         abp_flat_peaks.append(temp_peak)        
    #         abp_flat_lines.append(temp_line)
    #         number_of_bvalue_peak.append(temp_peak.count(True)/len(temp_peak))
    #         number_of_bvalue_line.append(temp_line.count(True)/len(temp_line))
            
    #     return number_of_bvalue_peak, number_of_bvalue_line, abp_flat_peaks, abp_flat_lines
###############################################################################
###############################################################################
############################################################################### 
    def detect_flat(self, pleth, abp, edge_flat, edge_peak):
        abp_flat_lines, abp_flat_peaks = [], []

        for sub in range(0,len(abp)):
            print("Subject number: "+str(sub+1))
            data = abp[sub]
            sub_flat_lines, sub_flat_peaks = [], []
            
            for cycle in range(0, len(data)):
                temp_line, temp_peak = [], []
                for sample in range(0, len(data[cycle])):
                    
                    data_max = argrelextrema(data[cycle], np.greater)
                    if sample in data_max[0]:   
                        temp_line.append(False)
                        if sample == len(data)-3:
                            if data[sample] == data[sample+1] and data[sample] == data[sample+2] and data[sample] == data[sample-1] :
                                temp_peak.append(True)
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_peak.append(False)
                            #     temp_line.append(False)                                
                        elif sample == len(data)-2:
                            if data[sample] == data[sample+1] and data[sample] == data[sample-1] and data[sample] == data[sample-2]:
                                temp_peak.append(True)
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_peak.append(False)
                            #     temp_line.append(False)
                        elif sample == len(data)-1:
                            if data[sample] == data[sample-1] and data[sample] == data[sample-2] and data[sample] == data[sample-3]:
                                temp_peak.append(True)
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_peak.append(False)  
                            #     temp_line.append(False)
                        else:
                            if data[sample] == data[sample+1] and data[sample] == data[sample+2] and data[sample] == data[sample+3]:
                                temp_peak.append(True)
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_peak.append(False)
                            #     temp_line.append(False)
                    else:
                        temp_peak.append(False)
                        if sample == len(data)-3:
                            if data[sample] == data[sample+1] and data[sample] == data[sample+2] and data[sample] == data[sample-1]:
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_line.append(False)
                        elif sample == len(data)-2:
                            if data[sample] == data[sample+1] and data[sample] == data[sample-1] and data[sample] == data[sample-2]:
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_line.append(False)  
                        elif sample == len(data)-1:
                            if data[sample] == data[sample-1] and data[sample] == data[sample-2] and data[sample] == data[sample-3]:
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_line.append(False)        
                        else:
                            if data[sample] == data[sample+1] and data[sample] == data[sample+2] and data[sample] == data[sample+3]:
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_line.append(False)
                number_of_bvalue_peak_sub, number_of_bvalue_line_sub = [], []   
                sub_flat_lines.append(temp_line)
                sub_flat_peaks.append(temp_peak)
                number_of_bvalue_peak_sub.append(sub_flat_lines.count(True)/len(sub_flat_lines))
                number_of_bvalue_line_sub.append(sub_flat_peaks.count(True)/len(sub_flat_peaks))
                
                checklist_lines, checklist_peak = [], []
                
                if sub_flat_lines.count(True)/len(sub_flat_lines) > edge_flat:
                    checklist_lines.append(False)
                else:
                    checklist_lines.append(True)  
                    
                    
                if sub_flat_peaks.count(True)/len(sub_flat_peaks) > edge_peak:
                    checklist_peak.append(False)
                else:
                    checklist_peak.append(True)  
                    
            abp_flat_peaks.append(checklist_lines)        
            abp_flat_lines.append(sub_flat_peaks)
        
        result_pleth, result_abp = []
        for sub in range(0, len(pleth)):
            if abp_flat_lines[sub] == True and abp_flat_peaks[sub] == True:
                result_pleth.append(pleth[sub])
                result_abp.append(abp[sub])

        return np.array(result_pleth, dtype=object), np.array(result_abp) 
###############################################################################
###############################################################################
############################################################################### 
    def segment_cycles(self, data):
        peak_detect = ElgPeakDetection(data, self.abp)
        peak_idx, block_idx, abp_mod = peak_detect.process()
        
        x_all, y_all = []
        for sub in range(0, len(data)):
            data_sub = data[sub]
            block_sub = block_idx[sub]
            l_diff = len(data_sub)-len(abp_mod[sub])
            
            data_new = data_sub[int(l_diff/2):int(len(data_sub)-l_diff/2)]
            
            sub_x, sub_y = []
            for idx in block_sub:
                temp_x = data_new[idx[0]:idx[1]]
                temp_y = abp_mod[idx[0]:idx[1]]
                sub_x.append(temp_x)
                sub_y.append(temp_y)
                
            x_all.append(sub_x)
            y_all.append(sub_y)
            x_all = np.array(x_all)
            y_all = np.array(y_all)
            
        return x_all, y_all
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
        
        
        
    
    
    
    
    






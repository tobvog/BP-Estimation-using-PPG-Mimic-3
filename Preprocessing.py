import numpy as np
import pandas as pd

from hampel import hampel
from scipy.signal import argrelextrema, butter, sosfiltfilt

class Preprocessing_mimic3:
    def __init__(self, data):
        self.pleth = data[0,1]
        self.abp = data[1,1]
        self.fs = data[0,2]
###############################################################################
###############################################################################
###############################################################################
    def freqfilt(data, lowcut=0.5, highcut=8, order=4):
        fs = 125
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='band', output="sos") 
    
        result = []        
        for sub in range(0,len(data)):
            temp = sosfiltfilt(sos, data[sub])
            result.append(temp)
        result = np.array(result, dtype=object)
            
        return result
###############################################################################
###############################################################################
###############################################################################
    def scaling(self, data):
        mean = np.nanmean(data)
        var = np.nanvar(data)
        result = (data-mean)/var
       
        return result        
###############################################################################
###############################################################################
###############################################################################  
    def change_nan(self):
        data = self.pleth
        result = data
        
        for i in range(0,len(data)):
            if np.isnan(data[i]) == True and np.isnan(data[i+1]) == False:
                result[i] = int((result[i-1]+result[i+1])/2)
                
            elif np.isnan(data[i]) == True and np.isnan(data[i+1]) == True and np.isnan(data[i+2]) == False:
                result[i] = int((result[i-1]+result[i+2])/2)
                result[i+1] = int((result[i-1]+result[i+2])/2)
                
            elif np.isnan(data[i]) == True and np.isnan(data[i+1]) == True and np.isnan(data[i+2]) == True:
                result[i] = 0
                
        return result       
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
    def detect_flat(self, pleth, abp, edge_flat=0.1, edge_peak=0.05):
        abp_flat_lines, abp_flat_peaks = [], []

        for sub in range(0,len(abp)):
            print("Subject number: "+str(sub+1))
            data = abp[sub]
            sub_flat_lines, sub_flat_peaks = [], []
            
            for cycle in range(0, len(data)):
                temp_line, temp_peak = [], []
                for sample in range(0, len(data[cycle])):
                    #print(str(cycle)+"_"+str(sample))
                    data_max = argrelextrema(data[cycle], np.greater)
                    if sample in data_max[0]:   
                        temp_line.append(False)
                        if sample == len(data[cycle])-3:
                            if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2] and data[cycle][sample] == data[cycle][sample-1] :
                                temp_peak.append(True)
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_peak.append(False)
                            #     temp_line.append(False)                                
                        elif sample == len(data[cycle])-2:
                            if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample-1] and data[cycle][sample] == data[cycle][sample-2]:
                                temp_peak.append(True)
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_peak.append(False)
                            #     temp_line.append(False)
                        elif sample == len(data[cycle])-1:
                            if data[cycle][sample] == data[cycle][sample-1] and data[cycle][sample] == data[cycle][sample-2] and data[cycle][sample] == data[cycle][sample-3]:
                                temp_peak.append(True)
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_peak.append(False)  
                            #     temp_line.append(False)
                        else:
                            if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2] and data[cycle][sample] == data[cycle][sample+3]:
                                temp_peak.append(True)
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_peak.append(False)
                            #     temp_line.append(False)
                    else:
                        temp_peak.append(False)
                        if sample == len(data[cycle])-3:
                            if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2] and data[cycle][sample] == data[cycle][sample-1]:
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_line.append(False)
                        elif sample == len(data[cycle])-2:
                            if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample-1] and data[cycle][sample] == data[cycle][sample-2]:
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_line.append(False)  
                        elif sample == len(data[cycle])-1:
                            if data[cycle][sample] == data[cycle][sample-1] and data[cycle][sample] == data[cycle][sample-2] and data[cycle][sample] == data[cycle][sample-3]:
                                temp_line.append(True)
                                break
                            # else:
                            #     temp_line.append(False)        
                        else:
                            if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2] and data[cycle][sample] == data[cycle][sample+3]:
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
        
        result_pleth, result_abp = [], []
        for sub in range(0, len(pleth)):
            if abp_flat_lines[sub] == True and abp_flat_peaks[sub] == True:
                result_pleth.append(pleth[sub])
                result_abp.append(abp[sub])

        return np.array(result_pleth, dtype=object), np.array(result_abp)
    
    # def detect_flat(self, data, abp, edge_flat=0.1, edge_peak=0.05):
    #     abp_flat_lines, abp_flat_peaks = [], []
            
    #     for cycle in range(0, len(data)):
    #         temp_line, temp_peak = [], []
    #         for sample in range(0, len(data[cycle])):

    #             data_max = argrelextrema(data[cycle], np.greater)
    #             if sample in data_max[0]:   
    #                 temp_line.append(False)
    #                 if sample == len(data[cycle])-3:
    #                     if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2] and data[cycle][sample] == data[cycle][sample-1] :
    #                         temp_peak.append(True)
    #                         temp_line.append(True)
    #                         break
                              
    #                 elif sample == len(data[cycle])-2:
    #                     if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample-1] and data[cycle][sample] == data[cycle][sample-2]:
    #                         temp_peak.append(True)
    #                         temp_line.append(True)
    #                         break

    #                 elif sample == len(data[cycle])-1:
    #                     if data[cycle][sample] == data[cycle][sample-1] and data[cycle][sample] == data[cycle][sample-2] and data[cycle][sample] == data[cycle][sample-3]:
    #                         temp_peak.append(True)
    #                         temp_line.append(True)
    #                         break

    #                 else:
    #                     if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2] and data[cycle][sample] == data[cycle][sample+3]:
    #                         temp_peak.append(True)
    #                         temp_line.append(True)
    #                         break

    #             else:
    #                 temp_peak.append(False)
    #                 if sample == len(data[cycle])-3:
    #                     if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2] and data[cycle][sample] == data[cycle][sample-1]:
    #                         temp_line.append(True)
    #                         break

    #                 elif sample == len(data[cycle])-2:
    #                     if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample-1] and data[cycle][sample] == data[cycle][sample-2]:
    #                         temp_line.append(True)
    #                         break

    #                 elif sample == len(data[cycle])-1:
    #                     if data[cycle][sample] == data[cycle][sample-1] and data[cycle][sample] == data[cycle][sample-2] and data[cycle][sample] == data[cycle][sample-3]:
    #                         temp_line.append(True)
    #                         break
       
    #                 else:
    #                     if data[cycle][sample] == data[cycle][sample+1] and data[cycle][sample] == data[cycle][sample+2] and data[cycle][sample] == data[cycle][sample+3]:
    #                         temp_line.append(True)
    #                         break
   
    #             abp_flat_lines.append(temp_line)
    #             abp_flat_peaks.append(temp_peak)
                
    #         check_lines = abp_flat_lines.count(True)/len()
                
    #             # number_of_bvalue_peak_sub = sub_flat_lines.count(True)/len(sub_flat_lines))
    #             # number_of_bvalue_line_sub.append(sub_flat_peaks.count(True)/len(sub_flat_peaks))
                
    #             #checklist_lines, checklist_peak = [], []
                
    #             if sub_flat_lines.count(True)/len(sub_flat_lines) > edge_flat:
    #                 checklist_lines.append(False)
    #             else:
    #                 checklist_lines.append(True)  
                    
                    
    #             if sub_flat_peaks.count(True)/len(sub_flat_peaks) > edge_peak:
    #                 checklist_peak.append(False)
    #             else:
    #                 checklist_peak.append(True)  
                    
    #         abp_flat_peaks.append(checklist_lines)        
    #         abp_flat_lines.append(sub_flat_peaks)
        
    #     result_pleth, result_abp = [], []
    #     for sub in range(0, len(data)):
    #         if abp_flat_lines[sub] == True and abp_flat_peaks[sub] == True:
    #             result_pleth.append(pleth[sub])
    #             result_abp.append(abp[sub])

    #     return np.array(result_pleth, dtype=object), np.array(result_abp)
###############################################################################
###############################################################################
###############################################################################   
    def segment_cycles(self, pleth, abp, block_idx):
              
        # l_diff = len(pleth)-len(abp)
        # abp_new = pleth[int(l_diff/2):int(len(abp)-l_diff/2)]
        
        cycle, label = [], []
        for i in range(0, len(block_idx), 2):
            if i == len(block_idx)-1:
                break
            else:
                idx_start = int(block_idx[i,0])
                idx_end = int(block_idx[i,1])
                temp_x = pleth[idx_start:idx_end]
                temp_y = abp[idx_start:idx_end]
                cycle.append(temp_x)
                label.append(temp_y)
                
        return np.array(cycle, dtype=object), np.array(label, dtype=object)
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

import numpy as np
from scipy.signal import sosfiltfilt, medfilt
## @brief This class realizes the prepocessing of the data. 
## @details This class realizes the prepocessing of the data. Additionally this class is used to extract cycles for the blood pressure estimation by Slapnicar et al.

class Preprocessing_mimic3:
    def __init__(self, data, sos):
        ##
        # @brief        This constructor initalizes the DataGenerator object.
        # @param data   Input data with 3 channels: PPG data[0], Blood pressure data[1], sampling rate[2].
        # @param sos    Array of second-order filter coefficients.
        ##
        self.pleth = data[0]
        self.abp =data[1]
        self.fs = data[2]
        self.sos = sos
 
###############################################################################
###############################################################################
###############################################################################
    def get_obj(self):
        ## 
        # @brief    This method returns the object of the class.
        # @return   Object of this class.
        ##
        return self.pleth, self.abp, self.fs
    
###############################################################################
###############################################################################
###############################################################################
    def filt_freq(cls):
        ## @brief    This method filters the frequencies of the ppg object.
        ##
        cls.pleth = sosfiltfilt(cls.sos, cls.pleth)
        
###############################################################################
###############################################################################
###############################################################################
    def scaling(cls):
        ## @brief    This method standardize the ppg object.
        ##
        mean = np.nanmean(cls.pleth)
        std = np.nanstd(cls.pleth)
        cls.pleth = (cls.pleth-mean)/std
       
###############################################################################
###############################################################################
###############################################################################  
    def change_nan(cls):
        ## @brief    This method change NaN values of the ppg object to mean (less than 3 in a row) or zero.
        ##
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
    def filt_median(cls, k_size=3):
        ##
        # @brief            This method applies a median filter to the ppg object.
        # @param k_size     Window size of the median filter. Default=3
        ##
        cls.pleth = medfilt(cls.pleth, k_size)    
        
###############################################################################
###############################################################################
###############################################################################     
    def detect_flat(self, data, edge_lines=0.1, edge_peaks=0.05):
        ##
        # @brief            This method detects same values (flat propertys) in a row. 
        # @param data       2 channel array with the cycles of ppg[0] and blood pressure[1].
        # @param edge_line  Limit of percentage of ordinary detected flats.
        # @param edge_peaks Limit of percentage of detected flats as peaks.
        # @return           Percentage of both flats and indices of cycles with flats. Additionally a bool value if subject passed or not.
        ##
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
        ##
        # @brief            This method segments cycles out of the timeseries. 
        # @param peak_idx   Indices of peaks of on subject.       
        # @param pad        Number of samples before and after peak for detect optimum cycle edges.
        # @return           Cycle of ppg and blood pressure signal.
        ##
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

###############################################################################
###############################################################################
###############################################################################
    def segment_cycles_nn(self, peak_idx, pleth):
        ##
        # @brief            This method segments cycles out of the timeseries for the resnet training of Slapnicar blood pressure estimation. 
        # @param peak_idx   Indices of peaks of on subject.       
        # @param pleth      PPG timeseries of one subject.
        # @return           Cycles of on subject and related peaks. 
        ##
        t = int(self.fs*2.5)
        dev1 = np.diff(pleth)
        dev2 = np.diff(dev1)
        
        dev0_cyc, dev1_cyc, dev2_cyc, peak_idx_nn = [[] for x in range(0, 3)]
        
        for nr_peak, idx_p in enumerate(peak_idx):
            try:
                dev0_cyc.append(pleth[idx_p-t:idx_p+t])
                dev1_cyc.append(dev1[idx_p-t:idx_p+t])
                dev2_cyc.append(dev2[idx_p-t:idx_p+t])
                peak_idx_nn.append(peak_idx[nr_peak])
            except:
                continue
            
        return np.array(dev0_cyc), np.array(dev1_cyc), np.array(dev2_cyc), np.array(peak_idx_nn) 
            
        
        
        
    
    
    
    
    
import numpy as np
from scipy.signal import argrelextrema
 
class ElgPeakDetection:
    def __init__(self, data, abp, w1_size=111, w2_size=667, fs=125, a=0.02, hampel_window=3):
        self.data = data
        self.abp = abp
        self.fs = fs
        self.a = a
        self.hampel_window = hampel_window
        self.w1 = int(w1_size*fs*10**(-3))
        self.w2 = int(w2_size*fs*10**(-3))
###############################################################################
###############################################################################
###############################################################################   
    def squaring(self):
        result = []
        for sample in range(0, len(self.data)):
            if self.data[sample] >= 0:
                temp = self.data[sample]**2
            else:
                temp = 0
            result.append(temp)
            
        return np.array(result, dtype=object)        
###############################################################################
###############################################################################
###############################################################################    
    def moving_average(self, data, window):
        if window == "w1":
            l = int(self.w1/2)
        elif window == "w2":
            l = int(self.w2/2)
        
        result = [] 
        for sample in range(l+1, len(data)-l):
            mean = np.mean(data[sample-l:sample+l])
            result.append(mean) 
        
        result = np.array(result, dtype=object)
        
        return result, l
###############################################################################
###############################################################################
###############################################################################        
    def correct_length(self, square, ma_peak, l_w1, l_w2):
        size_pleth = len(self.data)
        size_ma_peak = len(ma_peak)
        size_abp = len(self.abp)
        size_square = len(square)
        
        pleth_mod = self.data[l_w2:size_pleth-l_w2].to_numpy()
        ma_peak_mod = ma_peak[l_w2-l_w1:size_ma_peak-l_w2+l_w1]
        abp_mod = self.abp[l_w2-l_w1:size_abp-l_w2+l_w1]
        square_mod = square[l_w2:size_square-l_w2].to_numpy()
        
        return pleth_mod, abp_mod, ma_peak_mod, square_mod       
###############################################################################
###############################################################################
###############################################################################   
    def boi(self, data, ma_peak, ma_beat):
        z = np.mean(data)
        result = []
        ma_diff = int((len(ma_peak)-len(ma_beat))/2)
        for sample in range(ma_diff, len(ma_peak)-ma_diff):
            alpha = self.a*z
            thr1 = ma_beat[sample-ma_diff-1]+alpha
            if ma_peak[sample] > thr1:
                result.append(0.1)
            else:
                result.append(0)
                
        return np.array(result, dtype=object)        
###############################################################################
###############################################################################
###############################################################################      
    def boi_onset_offset(self, boi, data):
        def find_nearest(array, value):
            a = list(array)
            return min(range(len(a)), key=lambda i: abs(a[i]- value))
        
        thr2 = self.w1
        print(thr2)
        stat = True
        result, peaks = [], []
        min_idx = np.squeeze(np.array(argrelextrema(data, np.less)))
        
        for sample in range(0, len(boi)-1):
            if boi[sample] == 0 and boi[sample+1] == 0.1 and stat == True:
                idx_nearest = find_nearest(min_idx, sample+1)
                if min_idx[idx_nearest] < sample+1:
                    if idx_nearest+1 <   len(min_idx):
                        x1 = min_idx[idx_nearest]
                        x2 = min_idx[idx_nearest+1]
                        stat = False
                    else:
                        break
                else:
                    x1 = min_idx[idx_nearest-1]
                    x2 = min_idx[idx_nearest]
                    stat = False

                
            if stat == False:              
                if x2-x1 > thr2:
                    temp = data[x1:x2]
                    max_idx = np.argmax(temp)
                    peaks.append(max_idx+x1)
                               
                    result.append([x1, x2])
                    stat = True
                else:
                    stat = True
                         
        return np.array(result, dtype=object), np.array(peaks, dtype=object)
###############################################################################
###############################################################################
###############################################################################   
    def process(self):
        data_square = self.squaring()
        
        ma_peak, l_w1 = self.moving_average(data_square, self.w1)
        ma_beat, l_w2 = self.moving_average(data_square, self.w2)
        
        data_square_mod, ma_peak_mod, abp_mod = self.correct_length(self.data, self.abp, ma_peak, l_w1, l_w2)
        
        boi = self.boi(data_square_mod, ma_peak_mod, ma_beat)
        
        idx_blocks, idx_peaks = self.boi_onset_offset(boi, data_square_mod)
        
        return idx_blocks, idx_peaks, abp_mod

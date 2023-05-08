import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

 
class ElgPeakDetection:
    def __init__(self, data, w1_size=111, w2_size=667, a=0.02, hampel_window=3):
        self.pleth = data[0]
        self.abp = data[1]
        self.fs = data[2]
        self.a = a
        self.hampel_window = hampel_window
        self.w1 = int(w1_size*self.fs*10**(-3))
        self.w2 = int(w2_size*self.fs*10**(-3))
        self.pleth_squared = self.squaring(self.pleth)
###############################################################################
###############################################################################
###############################################################################   
    @staticmethod
    def squaring(pleth):
        result = []
        for sample in range(0, len(pleth)):
            if pleth[sample] >= 0:
                temp = pleth[sample]**2
            else:
                temp = 0
            result.append(temp)
            
        return np.array(result, dtype=object)      
###############################################################################
###############################################################################
###############################################################################    
    @staticmethod 
    def moving_average(self, window):
        if window == "w1":
            l = int(self.w1/2)
        elif window == "w2":
            l = int(self.w2/2)
        
        result = [] 
        for sample in range(l+1, len(self.pleth_squared)-l):
            mean = np.mean(self.pleth_squared[sample-l:sample+l])
            result.append(mean) 
        
        result = np.array(result, dtype=object)
        
        return result, l
###############################################################################
###############################################################################
############################################################################### 
    @staticmethod      
    def correct_length1(self, ma_peak, l_w1, l_w2):
        size_pleth = len(self.pleth)
        size_ma_peak = len(ma_peak)
        size_abp = len(self.abp)
        size_square = len(self.pleth_squared)
        
        pleth_mod = self.pleth[l_w2:size_pleth-l_w2]
        ma_peak_mod = ma_peak[l_w2-l_w1:size_ma_peak-l_w2+l_w1]
        abp_mod = self.abp[l_w2-l_w1:size_abp-l_w2+l_w1]
        square_mod = self.pleth_squared[l_w2:size_square-l_w2]
        
        return pleth_mod, abp_mod, ma_peak_mod, square_mod       
###############################################################################
###############################################################################
###############################################################################        
    def correct_length2(self, square, ma_peak, l_w1, l_w2):
        size_ma_peak = len(ma_peak)
        size_data = len(self.data)
        size_square = len(square)
        
        data_mod = self.data[l_w2:size_data-l_w2]
        ma_peak_mod = ma_peak[l_w2-l_w1:size_ma_peak-l_w2+l_w1]
        square_mod = square[l_w2:size_square-l_w2]
        
        return data_mod, ma_peak_mod, square_mod     
###############################################################################
###############################################################################
###############################################################################   
    def boi(self, pleth, ma_peak, ma_beat):
        z = np.mean(pleth)
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
    @staticmethod
    def boi_onset_offset(self, boi, data):
        def find_nearest(array, value):
            a = list(array)
            try:
                return min(range(len(a)), key=lambda i: abs(a[i]- value))
            except:
                return 0
               
        thr2 = self.w1
        stat = True
        result, peaks = [], []

        for sample in range(0, len(boi)-1):
            if boi[sample] == 0 and boi[sample+1] == 0.1 and stat == True:
                x1 = sample
                stat = False
                ct = 0
                while(stat==False):
                    ct += 1
                    if sample+ct >= len(boi):
                        stat=True
                        continue
                    
                    if boi[sample+ct]==0:
                        stat = True
                        x2 = sample+ct
                        if x2-x1>thr2:
                            temp = data[x1:x2]
                            max_idx = np.where(temp==temp.max())
                            peaks.append(x1+max_idx[0][0])
                            result.append([x1, x2]) 
                            stat=True
                        else:
                            stat=True
                         
        return np.array(result, dtype=object), np.array(peaks, dtype=object)
###############################################################################
###############################################################################
###############################################################################   
    def process(self):        
        ma_peak, l_w1 = self.moving_average(self,window="w1")
        ma_beat, l_w2 = self.moving_average(self,window="w2")
        
        pleth_mod, abp_mod, ma_peak_mod, pleth_square_mod = self.correct_length1(self, ma_peak, l_w1, l_w2)
        
        boi = self.boi(pleth_square_mod, ma_peak_mod, ma_beat)
        
        idx_blocks, idx_peaks = self.boi_onset_offset(self, boi, pleth_square_mod)
        
        return idx_blocks, idx_peaks, pleth_mod, abp_mod

import numpy as np


class ElgPeakDetection:
    
    def __init__(self, pleth, abp, w1_size=111, w2_size=667, fs=125, a=0.02, hampel_window=3):
        self.data = pleth
        self.abp = abp
        self.w1 = int(self.w1_size*self.fs*10**(-3)) 
        self.w2 = int(self.w2_size*self.fs*10**(-3))
        self.fs = fs
        self.a = a
        self.hampel_window = hampel_window
###############################################################################
###############################################################################
###############################################################################
    def squaring(self):
        result = []
        for sub in range(0, len(self.data)):
            temp_sub = []
            for sample in range(0, len(self.data[sub])):
                if self.data[sub][sample] >= 0:
                    temp = self.data[sub][sample]**2
                else:
                    temp = 0
                temp_sub.append(temp)
            result.append(temp_sub)

        result = np.array(result, dtype=object)
        
        return result
###############################################################################
###############################################################################
###############################################################################  
    def moving_average(self, data, window):
        l = int(window/2)
        
        result = []
        for sub in range(0, len(data)):
            temp_sub = []
            print("Subject Nr.: "+str(sub+1))
            for sample in range(l+1, len(data[sub])-l):
                mean = np.mean(data[sub][sample-l:sample+l])
                temp_sub.append(mean)    
            result.append(temp_sub)
            
        np.array(result, dtype=object)
        
        return result, l
###############################################################################
###############################################################################
###############################################################################     
    def correct_length(pleth, abp, ma_peak, l_w1, l_w2):
        pleth_mod, ma_peak_mod, abp_mod = [], [], []
        for sub in range(0,len(pleth)):
            size_pleth = len(pleth[sub])
            size_ma_peak = len(ma_peak[sub])
            size_abp = len(abp[sub])
            
            pleth_mod.append(pleth[sub][l_w2:size_pleth-l_w2])
            ma_peak_mod.append(ma_peak[sub][l_w2-l_w1:size_ma_peak-l_w2+l_w1])
            abp_mod.append(abp[sub][l_w2-l_w1:size_abp-l_w2+l_w1])
            
        return pleth_mod, abp_mod, ma_peak_mod
###############################################################################
###############################################################################
###############################################################################       
    def boi(self, data, ma_peak, ma_beat):
        boi = []
        for sub in range(0, len(data)):
            z = np.mean(data[sub])
            temp_sub = []
            ma_diff = int((len(ma_peak[sub])-len(ma_beat[sub]))/2)
            print("Subject Nr.: "+str(sub+1))
            
            for sample in range(ma_diff, len(ma_peak[sub])-ma_diff):
                alpha = self.a*z
                thr1 = ma_beat[sub][sample-ma_diff-1]+alpha
                if ma_peak[sub][sample] > thr1:
                    temp_sub.append(0.1)
                else:
                    temp_sub.append(0)
            
            boi.append(temp_sub)
        result = np.array(boi, dtype=object)
        
        return result
###############################################################################
###############################################################################
###############################################################################    
    def boi_onset_offset(self, boi, data):
        blocks = []
        w1 = int(self.wboi_size1*self.fs*10**(-3))   # 111 = w_size1     -> in ms
        #l_w1 = int(w1/2)
        #w2 = int(self.wboi_size2*self.fs*10**(-3))
        #l_w2 = int(w2/2)
        thr2 = w1
        peak_all = []
        for sub in range(0, len(boi)):
            size = len(data[sub])-len(boi[sub])
            data_temp = data[sub][size:] 
            
            print("Subject Nr.: "+str(sub+1))
            temp_sub, temp_peak = [], []
            x1, x2 = "x", "x"
            for sample in range(0, len(boi[sub])-1):
                if boi[sub][sample] == 0 and boi[sub][sample+1] == 0.1:
                    x1 = sample+1 
                                
                if boi[sub][sample] == 0.1 and boi[sub][sample-1] == 0.1:
                    x2 = sample
                    
                if x1 != "x" and x2 != "x" and x2-x1 > thr2:
                    if x2-x1 > thr2:
                        temp = data_temp[x1:x2]
                        max_idx = np.argmax(temp)
                        temp_peak.append(max_idx+x1)
                                       
                        temp_sub.append([x1, x2])
                        x1, x2 = "x", "x"
                    
            blocks.append(temp_sub)
            peak_all.append(temp_peak)
            
        blocks = np.array(blocks, dtype=object)    
        peak_all = np.array(peak_all, dtype=object)
        
        return peak_all, blocks
###############################################################################
###############################################################################
###############################################################################   
    def process(self):
        data_square = self.squaring()
        
        ma_peak, l_w1 = self.moving_average(data_square, self.w1)
        ma_beat, l_w2 = self.moving_average(data_square, self.w2)
        
        data_square_mod, ma_peak_mod, abp_mod = self.correct_length(self.abp, ma_peak, l_w1, l_w2)
        
        boi = self.boi(data_square_mod, ma_peak_mod, ma_beat)
        
        idx_blocks, idx_peaks = self.boi_onset_offset(boi, data_square_mod)
        
        return idx_blocks, idx_peaks
        
                
    
    
    
    
    
    
    
    
    
import numpy as np
## 
# @brief This class realizes a peak detection.  
# @detail For detailed information: https://pubmed.ncbi.nlm.nih.gov/24167546/
##
class ElgPeakDetection:
    def __init__(self, data, w1_size=111, w2_size=667, a=0.02):
        ## 
        # @brief            This constructor initalizes the class.
        # @param data       Input data with 3 channels: PPG data, Blood pressure data, sampling rate
        # @param w1_size    Window size in ms for systolic peak duration. Default=111. 
        # @param w2_size    Window size in ms for heart beat duration. Default=667.
        # @param a          Parameter for optimizing offset. Default=0.02.
        ##
        self.pleth = data[0]
        self.abp = data[1]
        self.fs = data[2]
        self.a = a
        
        self._w1 = int(w1_size*self.fs*10**(-3))
        self._w2 = int(w2_size*self.fs*10**(-3))
        self._pleth_squared = self.__squaring(self.pleth)
###############################################################################
###############################################################################
###############################################################################   
    @staticmethod
    def __squaring(pleth):
        ## 
        # @brief            This method squares the value of the input samples or set them to zero if they are less than zero.
        # @param pleth      PPG input data.
        # @return           Squared data as array.
        ##
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
    def __moving_average(self, window):
        ## 
        # @brief            This method realize a moving average filter.
        # @param window     Window size of the filter.
        # @return           Filtered data and half of window length.
        ##
        if window == "w1":
            l = int(self._w1/2)
        elif window == "w2":
            l = int(self._w2/2)
        
        result = [] 
        for sample in range(l+1, len(self._pleth_squared)-l):
            mean = np.mean(self._pleth_squared[sample-l:sample+l])
            result.append(mean) 
        
        result = np.array(result, dtype=object)
        
        return result, l
###############################################################################
###############################################################################
###############################################################################       
    def __correct_length(self, ma_peak, l_w1, l_w2):
        ## 
        # @brief            This method unifies the length of the input arrays
        # @param ma_peak    Moving average filtered peak signal.
        # @param l_w1       Length of window one.
        # @param l_w2       Length of window two.
        # @return           Length corrected data.
        ##
        size_pleth = len(self.pleth)
        size_ma_peak = len(ma_peak)
        size_abp = len(self.abp)
        size_square = len(self._pleth_squared)
        
        pleth_mod = self.pleth[l_w2:size_pleth-l_w2]
        ma_peak_mod = ma_peak[l_w2-l_w1:size_ma_peak-l_w2+l_w1]
        abp_mod = self.abp[l_w2-l_w1:size_abp-l_w2+l_w1]
        square_mod = self._pleth_squared[l_w2:size_square-l_w2]
        
        return pleth_mod, abp_mod, ma_peak_mod, square_mod    
###############################################################################
###############################################################################
###############################################################################   
    def __boi(self, pleth, ma_peak, ma_beat):
        ## 
        # @brief            This method extract boxes of interest as the value 0.1 and no boxes of interest as 0.0. 
        # @param pleth      Input ppg data.
        # @param ma_peak    Moving average filtered peak signal.
        # @param ma_beat    Moving average filtered beat signal.
        # @return           Boxes of interest as array
        ##
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
    def __boi_onset_offset(self, boi, data):
        ## 
        # @brief            This method extract the start and end indices of the boxes of interest 
        # @param boi        Boxes of interest marked as 0.1 and no bixes of interest marked as 0.0.
        # @param data       Preprocessed ppg data as input.
        # @return           Boxes of interest and peaks as array.
        ##
        def find_nearest(array, value):
            a = list(array)
            try:
                return min(range(len(a)), key=lambda i: abs(a[i]- value))
            except:
                return 0
               
        thr2 = self._w1
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
        ## 
        # @brief            Main method of the class to detect peaks.
        # @return           Indices of boxes of interest and peaks. Also the length corrected ppg and blood pressure signals.
        ##
        ma_peak, l_w1 = self.__moving_average(window="w1")
        ma_beat, l_w2 = self.__moving_average(window="w2")
        
        pleth_mod, abp_mod, ma_peak_mod, pleth_square_mod = self.__correct_length(ma_peak, l_w1, l_w2)
        
        boi = self.__boi(pleth_square_mod, ma_peak_mod, ma_beat)
        
        idx_blocks, idx_peaks = self.__boi_onset_offset(boi, pleth_square_mod)
        
        return idx_blocks, idx_peaks, pleth_mod, abp_mod

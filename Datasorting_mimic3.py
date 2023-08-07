import numpy as np
import os 
import wfdb

class Datasorting_mimic3:
    '''! Class for segmenting relevant data in respect to the paper of Slapnicar'''
    def __init__(self, path1="D:/MIMICIII_Database/mimic3wdb-1.0.physionet.org/"):
        ## 
        # @brief This constructor initalizes the class.
        # @param path1 Path of the Mimic3 Database
        ##
        self.path1=path1

    def check_existence(self, path2):
        ## 
        # @brief        This constructor initalizes the class.
        # @param path2  Deeper path of the Mimic3 Database
        # @return       Bool value of the input path existence.
        ##  
        return os.path.exists(self.path1+path2)
       
    def check_data_size(self, path2, min_b=17):
        ## 
        # @brief            This method checks the minimum number of bytes.
        # @param path2      Actual path of the target data.
        # @param min_b      Minimum number of kilobytes. 
        # @return result    Bool value if data is big enough.
        ##       
        size = round(os.path.getsize(self.path1+path2)/1024) # Datasize in kB
        if size > min_b:
            result = True
        else:
            result = False
            
        return result
                          
    def check_pleth_abp(self, path2):
        ## 
        # @brief            This method checks for the presence of blood pressure and ppg data.
        # @param path2      Actual path of the target data. 
        # @return result    Bool value if blood pressure and ppg data are available.
        ## 
        head = wfdb.rdheader(self.path1+path2)
        if "PLETH" in head.sig_name and "ABP" in head.sig_name:
            result = True
        else:
            result = False
        
        return result
      
    def check_time_length(self, path2):
        ## 
        # @brief            This method checks the time length of the data
        # @param path2      Actual path of the target data. 
        # @return result    Bool value if the time length is long enough.
        ## 
        data = wfdb.rdrecord(self.path1+path2)
        result = data.sig_len/(data.fs*60)
        
        return result
   
    def extract_data(self, path2):
        ## 
        # @brief            This method extract the target data.
        # @param path2      Actual path of the target data. 
        # @return result    Data which includes the target biosignals.
        ## 
        data = wfdb.rdrecord(self.path1+path2)
        idx_pleth = data.sig_name.index('PLETH')
        idx_abp = data.sig_name.index('ABP')
        
        pleth = np.array((data.record_name+'pleth', data.p_signal[:,idx_pleth], data.fs), dtype=object) 
        abp = np.array((data.record_name+'abp', data.p_signal[:,idx_abp], data.fs), dtype=object)
        
        result = np.array((pleth, abp))
        
        return result
        
        

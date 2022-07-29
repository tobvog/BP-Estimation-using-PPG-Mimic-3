import numpy as np
import os 
import wfdb


class Datasorting_mimic3:
    def __init__(self, path1="D:/MIMICIII_Database/mimic3wdb-1.0.physionet.org/"):
        self.path1=path1
        
    def check_existence(self, path2):
        result = os.path.exists(self.path1+path2)
        
        return result
        
    def check_data_size(self, path2):
        size = round(os.path.getsize(self.path1+path2)/1024) # Datasize in kB
        if size > 17:
            result = True
        else:
            result = False
            
        return result
                    
    def check_pleth_abp(self, path2):
        head = wfdb.rdheader(self.path1+path2)
        if "PLETH" in head.sig_name and "ABP" in head.sig_name:
            result = True
        else:
            result = False
        
        return result
    
    def check_time_length(self, path2):
        # time lenght in minutes
        data = wfdb.rdrecord(self.path1+path2)
        result = data.sig_len/(data.fs*60)
        
        return result

    def extract_data(self, path2):
        data = wfdb.rdrecord(self.path1+path2)
        idx_pleth = data.sig_name.index('PLETH')
        idx_abp = data.sig_name.index('ABP')
        
        pleth = np.array((data.record_name+'pleth', data.p_signal[:,idx_pleth], data.fs), dtype=object) 
        abp = np.array((data.record_name+'abp', data.p_signal[:,idx_abp], data.fs), dtype=object)
        
        result = np.array((pleth, abp))
        
        return result
        
        

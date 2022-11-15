## \brief Class for segmenting relevant data in respect to the paper of Slapnicar
##

import numpy as np
import os 
import wfdb

class Datasorting_mimic3:
    def __init__(self, path1="D:/MIMICIII_Database/mimic3wdb-1.0.physionet.org/"):
    ## \brief Initialization of the class
    # Need to be ckecked and changed if the "Festplatte" structure is different 
    #
    # \param[in]    path1      added path of the Mimic 3 Database 
    ##
        self.path1=path1

    def check_existence(self, path2):
    ## \brief Function for the checking of specific paths 
    # The data structure of the Mimic3 Database is not "einheitlich". This means the path need to be proofed by 
    # its existence.
    # 
    # \param[in]    path2      added path of specific subject of the Mimic 3 Database
    # \return       result     True if the path is existing, False if its not   
    ##
        result = os.path.exists(self.path1+path2)
        
        return result
       
    def check_data_size(self, path2):
    ## \brief Function for checking the number of bytes
    # The paper* works with data bigger than 17 kilobytes.
    #
    # \param[in]    path2      added path of specific subject of the Mimic 3 Database
    # \return       result     True if the data is big enough, False if its not 
    ## 
        size = round(os.path.getsize(self.path1+path2)/1024) # Datasize in kB
        if size > 17:
            result = True
        else:
            result = False
            
        return result
                          
    def check_pleth_abp(self, path2):
    ## \brief Function for checking if specific kind of signal is existence
    # The paper* works with photopletysmographie[ppg] and arterial blood pressure[abp] 
    # which existences need to checked.
    #
    # \param[in]    path2      added path of specific subject of the Mimic 3 Database
    # \return       result     True if ppg and abp are included, False if not
    ## 
        head = wfdb.rdheader(self.path1+path2)
        if "PLETH" in head.sig_name and "ABP" in head.sig_name:
            result = True
        else:
            result = False
        
        return result
      
    def check_time_length(self, path2):
    ## \brief Function for checking the length of the signal.
    # The paper* works with signals longer than 10 seconds which need to be checked 
    #
    # \param[in]    path2      added path of specific subject of the Mimic 3 Database
    # \return       result     Signal length in seconds 
    ## 
        data = wfdb.rdrecord(self.path1+path2)
        result = data.sig_len/(data.fs*60)
        
        return result
   
    def extract_data(self, path2):
    ## \brief Function to read .dat files
    # apb and ppg data is read by this function.
    #
    # \param[in]    path2      added path of specific subject of the Mimic 3 Database
    # \return       result     Matrix which contains the ppg and abp data 
    ##
        data = wfdb.rdrecord(self.path1+path2)
        idx_pleth = data.sig_name.index('PLETH')
        idx_abp = data.sig_name.index('ABP')
        
        pleth = np.array((data.record_name+'pleth', data.p_signal[:,idx_pleth], data.fs), dtype=object) 
        abp = np.array((data.record_name+'abp', data.p_signal[:,idx_abp], data.fs), dtype=object)
        
        result = np.array((pleth, abp))
        
        return result
        
        

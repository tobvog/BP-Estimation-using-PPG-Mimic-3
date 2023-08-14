# Imports
import numpy as np
import os
from scipy.signal import butter

from Preprocessing import Preprocessing_mimic3

# functions
def design_filt(fs=125, lowcut=0.5, highcut=8, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output="sos") 
    
    return sos

# Path of the first segmented data
path_main = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/segmented_data/"
# Target path
path_target = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/preprocessed_data"

ids = os.listdir(path_main) 
# Necessary subset of subject ids 
# ids = ids[:50]  

sos = design_filt()

if __name__=="__main__":
    for i in range(0,len(ids)):
        
        # Load data
        print('Loading File Number: '+str(i+1))
        data = np.load(path_main+ids[i], allow_pickle=True)
        data = [data[0,1], data[1,1], data[0,2]]
        
        # Initialize preprocessing class
        preprocess = Preprocessing_mimic3(data, sos)
        
        # Replace NaN values
        print("Step 1/4: change_nan")
        preprocess.change_nan()
        
        # Scaling
        print("Step 2/4: scaling")
        preprocess.scaling()
        
        # Frequency filter
        print("Step 3/4: filtering(frequency)")
        preprocess.filt_freq()
        
        # Median filter
        print("Step 4/4: filtering(median)")
        preprocess.filt_median()
        
        # Get object of the class
        pleth, abp, fs = preprocess.get_obj()
        
        print("|------------------------------|")
        # In order to save the data, the following should not be commented
        # np.save(path_target+ids[i], [pleth, abp, fs])







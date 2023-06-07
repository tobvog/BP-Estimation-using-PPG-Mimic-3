import numpy as np
import os

from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv1D, ReLU, BatchNormalization,Add, AveragePooling1D, Flatten, Dense, GRU, Concatenate
from tensorflow.keras.models import Model

def conv_block(inp, num_filter, kernel_size):
    outp = Conv1D(kernel_size=kernel_size, filters=num_filter)(inp)
    outp = BatchNormalization()(outp)
    outp = ReLU()(outp)
    
    return outp


def resnet_block(inp, num_filter=64, kernel_size=3):
    temp = conv_block(inp, num_filter, kernel_size)
    temp = conv_block(temp, num_filter, kernel_size)
    temp = conv_block(temp, num_filter, kernel_size)
    
    temp_short = Conv1D(kernel_size=kernel_size, filters=num_filter)(inp)
    temp_short = BatchNormalization()(temp_short)
    

    outp = Add()([temp, temp_short])
    outp = AveragePooling1D()(outp)
    
    return outp


def spec_temp_block(inp):
    outp = GRU()(inp)
    outp = BatchNormalization()(outp)
    
    return outp
    

def net_blocks(inp, size):
    inp = Input(size)
    outp_res = resnet_block(inp)
    outp_spec = spec_temp_block(inp)
    for i in range(0,3):
        outp_res = resnet_block(outp_res)
        outp_spec = spec_temp_block(outp_spec)
        
    return outp_res, outp_spec


def concat_blocks(res0, res1, res2, resblock):
    outp = Concatenate()([res0, res1, res2])
    if resblock==True:
        outp = GRU()(outp)
    outp = BatchNormalization()(outp)
    
    return outp
       
    
#%%
path_main = ""
files = os.listdir(path_main+"")

pleth0 = np.load()
pleth1 = np.load()
pleth2 = np.load()





    
    
    
    


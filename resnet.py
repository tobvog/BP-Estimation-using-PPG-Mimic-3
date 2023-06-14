import numpy as np
import os

from tensorflow.keras.layers import Input, Conv1D, Reshape, ReLU, BatchNormalization,Add, AveragePooling1D, Flatten, Dense, GRU, Concatenate, Permute, Dropout
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model 
from keras.regularizers import l2
from keras import optimizers
from kapre.time_frequency import Spectrogram
from kapre.utils import Normalization2D




def resnet_block(inp, num_filter, kernel_sizes=[8,5,3], pool_size=3, pool_stride_size=2):
    def conv_block(inp, num_filter, kernel_size):
        outp = Conv1D(kernel_size=kernel_size, filters=num_filter, padding="same")(inp)
        outp = BatchNormalization()(outp)
        outp = ReLU()(outp)
        
        return outp
    
    my_input = Input(shape=np.shape(inp))
    
    temp_long = inp
    temp_short = inp
    
    for i in range(0, len(kernel_sizes)):
        temp_long = conv_block(temp_long, num_filter, kernel_sizes[i])

    
    temp_short = Conv1D(kernel_size=1, filters=num_filter, padding="same")(temp_short)
    temp_short = BatchNormalization()(temp_short)
    

    outp = Add()([temp_long, temp_short])
    outp = AveragePooling1D(pool_size=pool_size, pool_stride_size=pool_stride_size)(outp)
    
    return my_input, outp


def spec_temp_block(inp):
    outp = Permute((2, 1))(inp)
    outp = Spectrogram(n_dft=128, n_hop=64, image_data_format="channels_last", return_decibel_spectrogram=True)(outp)
    outp = Normalization2D(str_axis="batch")(outp)
    outp = Reshape((2, 64))(outp)
    outp = GRU(64)(outp)
    outp = Flatten()(outp)
    outp = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(outp)
    outp = BatchNormalization()(outp)
    
    return outp
    

def net_blocks(inp, size):
    inp = Input(size)
    outp_res = resnet_block(inp, num_filter=64)
    outp_spec = spec_temp_block(inp)
    for i in range(0,4):
        outp_res = resnet_block(outp_res, num_filter=128)
        
    return outp_res, outp_spec


def concat_blocks(res0, res1, res2, resblock):
    outp = Concatenate()([res0, res1, res2])
    if resblock==True:
        outp = BatchNormalization()(outp)
        outp = GRU(65)(outp)
    outp = BatchNormalization()(outp)
    
    return outp
       
    
#%%
path_main = ""
files = os.listdir(path_main+"")

pleth0 = np.load()
pleth1 = np.load()
pleth2 = np.load()

l2_lambda = 0.001

inputs = [Input(shape=np.shape(pleth0)) for i in range(0,3)]

res_pleth0, spec_pleth0 = net_blocks(pleth0)
res_pleth1, spec_pleth1 = net_blocks(pleth1)
res_pleth2, spec_pleth2 = net_blocks(pleth2)

concat_resnet = concat_blocks(res_pleth0, res_pleth1, res_pleth2, True)
concat_spec = concat_blocks(spec_pleth0, spec_pleth1, spec_pleth2, False)

outp = Concatenate()([concat_resnet, concat_spec])
#outp = Flatten()(outp)
outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(outp)
outp = Dropout(0.25)(outp)
outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(outp) 
outp = Dropout(0.25)(outp)          
outp = Dense()(outp)    
final_outp = Dense(2, activation="relu")(outp)

model = Model(inputs=inputs, outputs=final_outp)
model = multi_gpu_model(model, gpus=2)
optimizer = optimizers.rmsprop(lr=.0001, decay=.0001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
print(model.summary())





    
    
    
    


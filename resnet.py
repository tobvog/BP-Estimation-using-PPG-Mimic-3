import numpy as np
import os
from DataGenerator import DataGenerator

from sklearn.model_selection import KFold, train_test_split


from tensorflow.keras.layers import Input, Conv1D, Reshape, LayerNormalization, ReLU, BatchNormalization,Add, AveragePooling1D, Flatten, Dense, GRU, concatenate, Permute, Dropout
from tensorflow.keras.models import Model
#from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model 
from keras.regularizers import l2
from keras import optimizers, Sequential
#from librosa.feature import melspectrogram
#from librosa.util import normalize
#import Spectrogram

import tensorflow as tf
#from kapre.time_frequency import Spectrogram
#from kapre.utils import Normalization2D

'''
class Spectrogram(keras.layers.Layer):
    def __init__(self, n_dft=256, n_hop=128, **kwargs):
        super(Spectrogram, self).__init__(**kwargs)
        self.n_dft = n_dft
        self.n_hop = n_hop

    def call(self, inputs):
        spectrogram = tf.signal.stft(inputs, frame_length=self.n_dft, frame_step=self.n_hop)
        spectrogram = tf.abs(spectrogram)
        return spectrogram

    def get_config(self):
        config = super(Spectrogram, self).get_config()
        config.update({'frame_length': self.n_dft, 'frame_step': self.n_hop})
        return config
'''
def resnet_block(inp, num_filter, kernel_sizes=[8,5,3], pool_size=3, pool_stride_size=2):
    def conv_block(inp, num_filter, kernel_size):
        outp = Conv1D(kernel_size=kernel_size, filters=num_filter, padding="same")(inp)
        outp = BatchNormalization()(outp)
        outp = ReLU()(outp)
        
        return outp
    
    outp_long = inp
    outp_short = inp
    
    for i in range(0, len(kernel_sizes)):
        outp_long = conv_block(outp_long, num_filter, kernel_sizes[i])
  
    outp_short = Conv1D(kernel_size=1, filters=num_filter, padding="same")(outp_short)
    outp_short = BatchNormalization()(outp_short)
    

    outp = Add()([outp_long, outp_short])
    outp = AveragePooling1D(pool_size=pool_size, strides=pool_stride_size)(outp)
    
    return outp


def spec_temp_block(inp):
    outp = Permute((2, 1))(inp)
    #outp = Spectrogram(n_dft=128, n_hop=64, image_data_format="channels_last", return_decibel_spectrogram=True)(outp)
    #outp = melspectrogram(n_dft=128, n_hop=64)(outp)
    #outp = Spectrogram(n_dft=128, n_hop=64)(outp)
    outp = tf.signal.stft(outp, frame_length=128, frame_step=64)
    outp = tf.abs(outp)
    #outp = normalize()(outp)
    #outp = Normalization2D(str_axis="batch")(outp)
    outp = LayerNormalization()(outp)
    outp = Reshape((2, 64))(outp)
    outp = GRU(64)(outp)
    outp = Flatten()(outp)
    outp = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(outp)
    outp = BatchNormalization()(outp)
    
    return outp
    

def net_blocks():
    inp_res = Input(shape=(None, 624))
    inp_spec = Input(shape=(None, 624))
    
    outp_res = resnet_block(inp_res, num_filter=64)
    outp_spec = spec_temp_block(inp_spec)
    for i in range(0,4):
        outp_res = resnet_block(outp_res, num_filter=128)
    
    model_time = Model(inp_res, outp_res)
    model_spec = Model(inp_spec, outp_spec)
        
    return model_time, model_spec, [inp_res, inp_spec]


def concat_blocks(inp0, inp1, inp2, resblock):
    outp = concatenate([inp0, inp1, inp2], axis=-1)
    #input_size = np.size(outp)
    
    if resblock==True:
        outp = BatchNormalization()(outp)
        outp = GRU(65)(outp)
    outp = BatchNormalization()(outp)
    
    #temp_model = Model(inputs=input_size, outputs = outp)
    #result_outp = temp_model(outp)
    
    return  outp
       
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)] 
'''        
def resnet(input_size, classes):
    #inputs = [Input(shape=np.shape(pleth0)) for i in range(0,3)]
    inputs = Input(input_size)
    
    res_pleth0, spec_pleth0 = net_blocks((1, input_size(1), input_size(2)))
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
    #model = multi_gpu_model(model, gpus=2)
    optimizer = optimizers.rmsprop(lr=.0001, decay=.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    print(model.summary())
    
'''        
def count_cyc(index, y):
    nr_cycle = [np.shape(y[i])[0] for i in range(0, len(y))]
    return sum(nr_cycle)



#%%
path_main = "E:/Uni/Master BMIT/Programmierprojekt/feat/"
files = os.listdir(path_main+"derivations/dev0")

#labels = np.array([np.load(path_main+"ground_truth/nn/"+subject, allow_pickle=True) for subject in files], dtype=object)
files = files[:25]

n_splits = 2
kfold = KFold(n_splits=n_splits)
kfold.get_n_splits(files)

batch_size = 1
l2_lambda = 0.001   

for train_index, test_index in kfold.split(files): 
    
    train_index, val_index = train_test_split(train_index, test_size=0.2)
    train_id = [files[x] for x in train_index]
    val_id = [files[x] for x in val_index]
    # nr_cyc_train = count_cyc(train_index, labels)
    # nr_cyc_val = count_cyc(val_index, labels)
    
    # Generators
    
    generator_dev0 = DataGenerator(path_main+"derivations/dev0/", train_id, path_main+"ground_truth/nn/", batch_size=batch_size)
    generator_dev1 = DataGenerator(path_main+"derivations/dev1/", train_id, path_main+"ground_truth/nn/", batch_size=batch_size)
    generator_dev2 = DataGenerator(path_main+"derivations/dev2/", train_id, path_main+"ground_truth/nn/", batch_size=batch_size)
    
    generator_val = DataGenerator(path_main+"derivations/dev0/", val_id, path_main+"ground_truth/nn/", batch_size=len(val_id))
    
    model_time_dev0, model_spec_dev0, inp_dev0 = net_blocks()
    model_time_dev1, model_spec_dev1, inp_dev1 = net_blocks()
    model_time_dev2, model_spec_dev2, inp_dev2 = net_blocks()
    
    
    '''
    outp0_time = model_time_dev0(generator_dev0)
    outp1_time = model_time_dev1(generator_dev1)
    outp2_time = model_time_dev1(generator_dev2)
    
    outp0_spec = model_spec_dev0(generator_dev0)
    outp1_spec = model_spec_dev1(generator_dev1)
    outp2_spec = model_spec_dev1(generator_dev2)
    '''
    outp_concat_resnet = concat_blocks(model_time_dev0.output, model_time_dev1.output, model_time_dev2.output, True)
    outp_concat_spec = concat_blocks(model_spec_dev0.output, model_spec_dev1.output, model_spec_dev2.output, False)
    
    merged = concatenate([outp_concat_resnet, outp_concat_spec], axis=-1)
    #outp = Flatten()(outp)
    #final_model = Model(inputs=[model_x1.input, model_x2.input, model_x3.input], outputs=merged_output)
    merged_outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(merged)
    merged_outp = Dropout(0.25)(merged_outp)
    merged_outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(merged_outp)
    merged_outp = Dropout(0.25)(merged_outp)
    merged_outp = Dense(2, activation="relu")(merged_outp)
    '''
    final_model = Model(inputs=[model_time_dev0.input, 
                                model_time_dev1.input, 
                                model_time_dev2.input, 
                                model_spec_dev0.input, 
                                model_spec_dev1.input,
                                model_spec_dev2.input],
                        outputs=merged_outp)
    '''
    final_model = Model(inputs=[inp_dev0[0], 
                                inp_dev1[0],
                                inp_dev2[0],
                                inp_dev0[1],
                                inp_dev1[1],
                                inp_dev2[1]],
                        outputs=merged_outp)
    '''
    final_model.add(Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda)))
    final_model.add(Dropout(0.25))
    final_model.add(Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda)))
    final_model.add(Dropout(0.25))
    final_model.add(Dense(2, activation="relu"))
    
    outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(outp)
    outp = Dropout(0.25)(outp)
    outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(outp) 
    outp = Dropout(0.25)(outp)          
    outp = Dense()(outp)    
    final_outp = Dense(2, activation="relu")(outp)
    final_model = Model(inputs=inputs, outputs=final_outp)
    #model = multi_gpu_model(model, gpus=2)
    '''
    
    optimizer = optimizers.RMSprop(lr=.0001)
    final_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    #print(final_model.summary())
    
    generator_input = np.asarray([generator_dev0, generator_dev1, generator_dev2])
    final_model.fit(generator_input, 
                    validation_data=generator_val,
                    epochs=20,
                    verbose=1)
    





    
    
    
    


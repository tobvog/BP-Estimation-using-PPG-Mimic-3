import numpy as np
import os
from DataGenerator_Outp2 import DataGenerator

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from tensorflow import  squeeze
from tensorflow.keras.layers import Input, Conv1D, Reshape, LayerNormalization, ReLU, BatchNormalization, Add, AveragePooling1D, Flatten, Dense, GRU, concatenate, Dropout 
from keras.regularizers import l2
from keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import tensorflow as tf

keras.backend.clear_session()

def resnet_block(inp, batch_size, num_filter, kernel_sizes=[8,5,3], pool_size=3, pool_stride_size=2):
    def conv_block(inp, batch_size, num_filter, kernel_size):
        outp = Conv1D(kernel_size=kernel_size, filters=num_filter, padding="same")(inp)
        outp = BatchNormalization()(outp)
        outp = ReLU()(outp)       
        return outp
    
    def create_output_long(inp, batch_size, num_filter, kernel_sizes):
        outp = inp
        for i in range(0, len(kernel_sizes)):
            outp = conv_block(outp, batch_size, num_filter, kernel_sizes[i])
        return outp
    
    def create_output_short(inp, num_filter):
        outp = Conv1D(kernel_size=1, filters=num_filter, padding="same")(inp)
        outp = BatchNormalization()(outp)
        return outp
    
    outp_long = create_output_long(inp, batch_size, num_filter, kernel_sizes)
    outp_short = create_output_short(inp, num_filter)

    outp = Add()([outp_long, outp_short])

    if len(outp.shape) == 4:     
        outp = squeeze(outp, axis=1)

    outp = AveragePooling1D(pool_size=pool_size, strides=pool_stride_size)(outp)
    
    return outp


def spec_temp_block(inp, batch_size):

    outp = tf.signal.stft(inp, frame_length=128, frame_step=64)
    outp = tf.abs(outp)
    outp = LayerNormalization()(outp)
    outp = Reshape((8*batch_size,65))(outp)
    outp = GRU(65)(outp)
    outp = Flatten()(outp)
    outp = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(outp)
    outp = BatchNormalization()(outp)
    
    return outp
   

def net_blocks(batch_size):
    inp_time = Input(shape=(batch_size, 624), name="input_time0")
    inp_spec = Input(shape=(batch_size, 624), name="input_spec0")

    
    outp_time = resnet_block(inp_time, batch_size=batch_size, num_filter=64)
    outp_spec = spec_temp_block(inp_spec, batch_size=batch_size)

    for i in range(0,4):
        outp_time = resnet_block(outp_time, batch_size=batch_size, num_filter=128)

    model_time = keras.Model(inp_time, outp_time)#, name=model_names[0])
    model_spec = keras.Model(inp_spec, outp_spec)#, name=model_names[0])
        
    return model_time, model_spec, inp_time, inp_spec, outp_time, outp_spec


def concat_blocks(inp0, inp1, inp2, resblock, l_name): 
    outp = concatenate([inp0, inp1, inp2], axis=-1)
    
    if resblock==True:
        outp = BatchNormalization()(outp)
        outp = GRU(384)(outp)
    outp = BatchNormalization()(outp)
    
    return  outp
      

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        


#path_main = "E:/Uni/Master BMIT/Programmierprojekt/feat2/"
path_main = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/feat_new/"
files = os.listdir(path_main+"derivations/dev0")
#labels = np.array([np.load(path_main+"ground_truth/nn/"+subject, allow_pickle=True) for subject in files], dtype=object)

n_splits = 5
kfold = KFold(n_splits=n_splits)
kfold.get_n_splits(files)
kernel_init = "he_uniform"

batch_size = 1028
l2_lambda = 0.001   

all_mae_sbp, all_mae_dbp = [], []
for train_index, test_index in kfold.split(files[:20]): 
    
    train_index, val_index = train_test_split(train_index, test_size=0.2)
    train_id = [files[x] for x in train_index]
    val_id = [files[x] for x in val_index]
    test_id = [files[x] for x in test_index]
    
    # Generators
    print("Loading Datagenerator")
    generator_train = DataGenerator(path_main, train_id, batch_size=batch_size)
    generator_val = DataGenerator(path_main, val_id, batch_size=batch_size)
    generator_test = DataGenerator(path_main, test_id, batch_size=batch_size)

    
    print("Creating Model")
    model_time, model_spec, input_time, input_spec, outp_time, outp_spec = net_blocks(batch_size)
    outp_time = Reshape((1, 3968))(outp_time)
    outp_time = squeeze(outp_time, axis=1)
    merged = concatenate([outp_time, outp_spec])
    merged_outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda), kernel_initializer=kernel_init)(merged)
    merged_outp = Dropout(0.2)(merged_outp)
    merged_outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda), kernel_initializer=kernel_init)(merged_outp)
    merged_outp = Dropout(0.2)(merged_outp)
    merged_outp = Dense(2, activation='relu')(merged_outp)

    final_model = keras.Model(inputs=[input_time,
                            input_spec],
                            outputs=merged_outp,
                            name='Final_Model')
    
    
    
    
    optimizer = optimizers.RMSprop(learning_rate=0.0001)
    
    es = EarlyStopping(monitor="mae", patience=5)
    final_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    # print(final_model.summary())
    
    final_model.fit(generator_train, 
                    validation_data=generator_val,
                    epochs=1,
                    verbose=1, 
                    callbacks=[es])
    
    # y_pred = final_model.predict(generator_test)
    # y_true = np.array([np.load(path_main+"ground_truth/nn/"+subject, allow_pickle=True) for subject in test_id])
    all_pred = []
    all_true = []
    for batch_index in range(generator_test.__len__()):
        print(batch_index)
        batch_data, batch_true_labels = generator_test.__getitem__(batch_index)

        batch_pred = final_model.predict(batch_data, verbose=0)
        all_pred.append(batch_pred)
        sbp_mean = np.mean(batch_true_labels[:,0])
        dbp_mean = np.mean(batch_true_labels[:,1])
        all_true.append(np.array([sbp_mean, dbp_mean]))

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.array(all_true)

    mae_sbp = mean_absolute_error(all_pred[:,0], all_true[:,0])
    mae_dbp = mean_absolute_error(all_pred[:,1], all_true[:,1])

    all_mae_sbp.append(mae_sbp)
    all_mae_dbp.append(mae_dbp)

mae_sbp_mean = np.mean(all_mae_sbp)
mae_dbp_mean = np.mean(all_mae_dbp)



    
    


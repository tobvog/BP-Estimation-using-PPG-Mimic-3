#---------------------------------------------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
from DataGenerator_Outp2 import DataGenerator

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from tensorflow import  squeeze
from tensorflow.keras.layers import Input, Conv1D, Concatenate, Permute, ReLU, BatchNormalization, Add, AveragePooling1D, Flatten, Dense, GRU, Dropout 
from keras.regularizers import l2
from keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
import tensorflow as tf

keras.backend.clear_session()

#---------------------------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------------------------
# ResNet Block
def resnet_block(inp, num_filter, kernel_sizes=[8,5,3], pool_size=2, pool_stride_size=1):
    def conv_block(inp, num_filter, kernel_size):
        outp = Conv1D(kernel_size=kernel_size, filters=num_filter, padding="same")(inp)
        outp = BatchNormalization()(outp)
        outp = ReLU()(outp)       
        return outp
    
    def create_output_long(inp, num_filter, kernel_sizes):
        outp = inp
        for i in range(0, len(kernel_sizes)):
            outp = conv_block(outp, num_filter, kernel_sizes[i])
        outp= BatchNormalization()(outp)
        return outp
    
    
    def create_output_short(inp, num_filter):
        outp = Conv1D(kernel_size=1, filters=num_filter, padding="same")(inp)
        outp = BatchNormalization()(outp)
        return outp
    
    outp_long = create_output_long(inp, num_filter, kernel_sizes)
    outp_short = create_output_short(inp, num_filter)
    outp = Add()([outp_long, outp_short])
    outp = ReLU()(outp)

    return outp

# Spectral Block
def spec_temp_block(inp):
    outp = squeeze(inp, axis=0) 
    outp = Permute((2,1))(outp)
    outp = tf.signal.stft(outp, frame_length=128, frame_step=64)
    outp = tf.abs(outp)
    outp = squeeze(outp, axis=1)
    outp = GRU(64, return_sequences=True)(outp)
    outp = Flatten()(outp)
    outp = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(outp)
    outp = BatchNormalization()(outp)

    return outp
   
# Create blocks of the network 
def net_blocks():
    inp_time = Input(shape=(None, 624, 1), name="input_time0")
    inp_spec = Input(shape=(None, 624, 1), name="input_spec0")
    
    outp_time = resnet_block(inp_time, num_filter=64) 
    outp_time = squeeze(outp_time, axis=0)
    outp_time = AveragePooling1D(pool_size=3, strides=2)(outp_time)
    outp_spec = spec_temp_block(inp_spec)

    for i in range(0,4):
        outp_time = resnet_block(outp_time, num_filter=128)
        if i < 3:
            outp_time = AveragePooling1D(pool_size=3, strides=2)(outp_time)
            
    outp_time = Flatten()(outp_time)
    model_time = keras.Model(inp_time, outp_time)
    model_spec = keras.Model(inp_spec, outp_spec)
        
    return model_time, model_spec, inp_time, inp_spec, outp_time, outp_spec

#---------------------------------------------------------------------------------------------------------------------
# Initialize paths
#---------------------------------------------------------------------------------------------------------------------

# Main path of final preprocessed data
path_main = "E:/Uni/Master BMIT/Programmierprojekt/feat2/"
# IDs
files = os.listdir(path_main+"derivations/dev0")
### Necessary for using subset ###
files = files[:10]
##################################

#---------------------------------------------------------------------------------------------------------------------
# Initiliaze trainingsparameter
#---------------------------------------------------------------------------------------------------------------------

# Initialize kfold
n_splits = 2
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
kfold.get_n_splits(files)

# Initialize Hyperparameter
kernel_init = "he_uniform"
batch_size = 256
l2_lambda = 0.001  

if __name__=="__main__":    
    
    all_mae_sbp, all_mae_dbp = [], []
    for nr_fold, (train_index, test_index) in enumerate(kfold.split(files)): 
        if nr_fold == 2:
            continue
        # Define all indices
        train_index, val_index = train_test_split(train_index, test_size=0.2)
        train_id = [files[x] for x in train_index]
        val_id = [files[x] for x in val_index]
        test_id = [files[x] for x in test_index]
        
        # Generators
        print("Loading Datagenerator")
        generator_train = DataGenerator(path_main, train_id, batch_size=batch_size)
        generator_val = DataGenerator(path_main, val_id, batch_size=batch_size)
        generator_test = DataGenerator(path_main, test_id, batch_size=batch_size)
    
        # Create Model
        print("Creating Model")
        model_time, model_spec, input_time, input_spec, outp_time, outp_spec = net_blocks()

        merged = Concatenate()([outp_time, outp_spec])        
        merged_outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda), kernel_initializer=kernel_init)(merged)
        merged_outp = Dropout(0.25)(merged_outp)
        merged_outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda), kernel_initializer=kernel_init)(merged_outp)
        merged_outp = Dropout(0.25)(merged_outp)
        merged_outp = Dense(2, activation='relu')(merged_outp)
        final_model = keras.Model(inputs=[input_time,
                                input_spec],
                                outputs=merged_outp,
                                name='Final_Model')
        
        
        # Create training
        optimizer = optimizers.RMSprop(learning_rate=0.0001)
        es = EarlyStopping(monitor="mae", patience=3)
        mcp = ModelCheckpoint('best_model'+str(nr_fold)+'.h5', monitor='val_loss', save_best_only=True)
        final_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
        final_model.fit(generator_train, 
                        validation_data=generator_val,
                        epochs=5,
                        verbose=1, 
                        callbacks=[es, mcp])
        
        # Make prediction
        nr_data = generator_test.__len__()
        y_pred = np.zeros((nr_data*batch_size,2))
        y_true = np.zeros((nr_data*batch_size,2))
        all_mae = np.zeros((nr_data*batch_size,2))
        print('Prediction')
        for batch_index in range(nr_data):
            batch_data, batch_true_labels = generator_test.__getitem__(batch_index)
            
            temp_pred = final_model.predict(batch_data, verbose=0, batch_size=batch_size)
            temp_true = batch_true_labels
            
            mae_sbp_batch = mean_absolute_error(temp_pred[:,0], temp_true[:,0])
            mae_dbp_batch = mean_absolute_error(temp_pred[:,1], temp_true[:,1])
            
            all_mae[batch_index] = np.array([mae_sbp_batch, mae_dbp_batch])
             
        mae_sbp = np.mean(all_mae[:,0])
        mae_dbp = np.mean(all_mae[:,1])
    
        all_mae_sbp.append(mae_sbp)
        all_mae_dbp.append(mae_dbp)
    
    mae_sbp_mean = np.mean(all_mae_sbp)
    mae_dbp_mean = np.mean(all_mae_dbp)
    
    print("Mean of SBP: ", mae_sbp_mean)
    print("Mean of DBP: ", mae_dbp_mean)
    
#%%

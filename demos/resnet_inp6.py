import numpy as np
import os
from DataGenerator.DataGenerator_Outp6 import DataGenerator

from sklearn.model_selection import KFold, train_test_split
from tensorflow import squeeze
from tensorflow.keras.layers import Input, Conv1D, Reshape, LayerNormalization, ReLU, BatchNormalization,Add, AveragePooling1D, Flatten, Dense, GRU, concatenate, Dropout 
from keras.regularizers import l2
from keras import optimizers
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
    inp_time0 = Input(shape=(batch_size, 624), name="input_time0")
    inp_spec0 = Input(shape=(batch_size, 624), name="input_spec0")
    inp_time1 = Input(shape=(batch_size, 624), name="input_time1")
    inp_spec1 = Input(shape=(batch_size, 624), name="input_spec1")
    inp_time2 = Input(shape=(batch_size, 624), name="input_time2")
    inp_spec2 = Input(shape=(batch_size, 624), name="input_spec2")
    
    outp_time0 = resnet_block(inp_time0, batch_size=batch_size, num_filter=64)
    outp_spec0 = spec_temp_block(inp_spec0, batch_size=batch_size)
    outp_time1 = resnet_block(inp_time1, batch_size=batch_size, num_filter=64)
    outp_spec1 = spec_temp_block(inp_spec1, batch_size=batch_size)
    outp_time2 = resnet_block(inp_time2, batch_size=batch_size, num_filter=64)
    outp_spec2 = spec_temp_block(inp_spec2, batch_size=batch_size)
    
    for i in range(0,4):
        outp_time0 = resnet_block(outp_time0, batch_size=batch_size, num_filter=128)
        outp_time1 = resnet_block(outp_time1, batch_size=batch_size, num_filter=128)
        outp_time2 = resnet_block(outp_time2, batch_size=batch_size, num_filter=128)
        
    model_time0 = keras.Model(inp_time0, outp_time0)#, name=model_names[0])
    model_spec0 = keras.Model(inp_spec0, outp_spec0)#, name=model_names[0])
    model_time1 = keras.Model(inp_time1, outp_time1)#, name=model_names[1])
    model_spec1 = keras.Model(inp_spec1, outp_spec1)#, name=model_names[1])
    model_time2 = keras.Model(inp_time2, outp_time2)#, name=model_names[2])
    model_spec2 = keras.Model(inp_spec2, outp_spec2)#, name=model_names[2])
        
    return [model_time0, model_time1, model_time2], [model_spec0, model_spec1, model_spec2], [inp_time0, inp_time1, inp_time2], [inp_spec0, inp_spec1, inp_spec2]


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


#path_main = "E:/Uni/Master BMIT/Programmierprojekt/feat/"
path_main = "C:/Users/vogel/Desktop/Study/Master BMIT/1.Semester/Programmierprojekt/feat_new/"
files = os.listdir(path_main+"derivations/dev0")
#labels = np.array([np.load(path_main+"ground_truth/nn/"+subject, allow_pickle=True) for subject in files], dtype=object)


n_splits = 5
kfold = KFold(n_splits=n_splits)
kfold.get_n_splits(files)
kernel_init = "he_uniform"

batch_size = 128
l2_lambda = 0.001   


for train_index, test_index in kfold.split(files): 
    
    train_index, val_index = train_test_split(train_index, test_size=0.2)
    train_id = [files[x] for x in train_index]
    val_id = [files[x] for x in val_index]
    
    # Generators
    generator_train = DataGenerator(path_main, train_id, path_main+"ground_truth/nn/", batch_size=batch_size)
    generator_val = DataGenerator(path_main, val_id, path_main+"ground_truth/nn/", batch_size=batch_size)
       
    models_time, models_spec, inputs_time, inputs_spec = net_blocks(batch_size)   
    outp_concat_resnet = concat_blocks(models_time[0].output, models_time[1].output, models_time[2].output, True, l_name="reshape_concat_resnet")
    outp_concat_spec = concat_blocks(models_spec[0].output, models_spec[1].output, models_spec[2].output, False, l_name="reshape_concat_spec")
    print(outp_concat_resnet.shape)  
    print(outp_concat_spec.shape) 
    merged = concatenate([outp_concat_resnet, outp_concat_spec], axis=-1)
    merged_outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda), kernel_initializer=kernel_init)(merged)
    merged_outp = Dropout(0.2)(merged_outp)
    merged_outp = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda), kernel_initializer=kernel_init)(merged_outp)
    merged_outp = Dropout(0.2)(merged_outp)
    merged_outp = Dense(2, activation='relu')(merged_outp)

    final_model = keras.Model(inputs=[inputs_time[0], 
                                inputs_time[1],
                                inputs_time[2],
                                inputs_spec[0],
                                inputs_spec[1],
                                inputs_spec[2]],
                        outputs=merged_outp,
                        name='Final_Model')
    
    optimizer = optimizers.RMSprop(learning_rate=0.0001)
    final_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    # print(final_model.summary())
    
    final_model.fit(generator_train, 
                    validation_data=generator_val,
                    epochs=20,
                    verbose=1)
#%%
def myprint(s):
    with open('modelsummary.txt','a') as f:
        print(s, file=f)

final_model.summary(print_fn=myprint)
#%%
for layer in final_model.layers:
    if layer.name == "input_time3":
        print(layer.name, layer.output_shape)
    elif layer.name == "conv1d_228":
        print(layer.name, layer.input_shape)
    else:
        print(layer.name)
#%% Show Input Shapes
print("Input shapes:")
for i, input_tensor in enumerate(final_model.inputs):
    print(f"Input {i}: {input_tensor.shape}")
#%% Show all layer shapes
for layer in final_model.layers:
    print(layer.name, layer.input_shape, layer.output_shape)
#%% Show specfic Input/Output
for layer in final_model.layers:
    if layer.name == "input_time3":
        print(layer.name, layer.output_shape)
    elif layer.name == "conv1d_843":
        print(layer.name, layer.input_shape)
    else:
        print(layer.name)


#%%


#for train_index, test_index in kfold.split(files): 
    
train_index, val_index = train_test_split(train_index, test_size=0.2)
train_id = [files[x] for x in train_index]
val_id = [files[x] for x in val_index]

# Generators
generator_train = DataGenerator(path_main, train_id, path_main+"ground_truth/nn/", batch_size=batch_size)
generator_val = DataGenerator(path_main, val_id, path_main+"ground_truth/nn/", batch_size=batch_size)
 


#%%
    
def data_manual(datagenerator):
  """Generates a data manual for the given datagenerator."""

  print("Datagenerator parameters:")
  for key, value in datagenerator.__dict__.items():
    if not key.startswith("_"):
      print(f"  {key}: {value}")

  print("\nData manual:")
  for idx in range(datagenerator.__len__()):
    x, y = datagenerator.__getitem__(idx)
    print(f"Batch {idx}:")
    print(f"  x: {x}")
    print(f"  y: {y}")


if __name__ == "__main__":
  datagenerator = generator_train
  data_manual(datagenerator)


#%%
# Set random seed for reproducibility (if you haven't already done it)
np.random.seed(42)
tf.random.set_seed(42)

# Create an instance of the data generator
# Replace the arguments with appropriate values
data_gen =  DataGenerator(path_main, train_id, path_main+"ground_truth/nn/", batch_size=batch_size)
# Run the generator multiple times and store the outputs
num_runs = 5
outputs = [data_gen.__getitem__(0) for _ in range(num_runs)]

# Compare the first output with the rest to check for consistency
consistent_output = all(np.array_equal(outputs[0][0], output[0]) and np.array_equal(outputs[0][1], output[1]) for output in outputs[1:])

if consistent_output:
    print("Generator produces the same output in multiple runs.")
else:
    print("Generator produces different outputs.")

    
    
    


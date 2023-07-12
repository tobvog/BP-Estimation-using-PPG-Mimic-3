import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataGenerator(keras.utils.Sequence):
    def __init__(self, path_main, list_id, path_label, batch_size, validation_split=0.2, shuffle=True):
        self.path_main = path_main
        self.path_label = path_label
        self.list_id = list_id
        self.batch_size = batch_size
        self.y, self.x = None, None
        # self.x_train, self.x_val, self.y_train, self.y_val =  [None for i in range(0,4)]
        # self.validation_split = validation_split
        self.shuffle = shuffle
        # self._split_data()
        'Initialization'
        '''
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        '''
    # def _split_data(self):
    #         self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.list_id,
    #                                                                               self.list_id,
    #                                                                           shuffle=self.shuffle)           
    # def use_validation(self):
    #     self.x = self.x_val
    #     self.y = self.y_val

    # def use_training(self):
    #     self.x = self.x_train
    #     self.y = self.y_train

    def __len__(self):
        return int(np.ceil(len(self.list_id) / float(self.batch_size)))
        # return len(self.y) // self.batch_size

    def __getitem__(self, idx):
        batch_id = self.list_id[idx*self.batch_size : (idx+1)*self.batch_size]

        # Load and process the input and target data batch
        input_data = []
        target_data = []
        
        for path in batch_id:
            # Load the input data from external storage
            input_data_batch = np.load(self.path_main+path, allow_pickle=True)
            target_data_batch = np.load(self.path_label+path)
            
            input_data_batch_mod, target_data_batch_mod = [], []
            for i in range(0, len(target_data_batch)):
                if len(input_data_batch[i]) == 624:
                    input_data_batch_mod.append(input_data_batch[i])
                    target_data_batch_mod.append(target_data_batch[i])
            
            input_data.append(input_data_batch_mod)
            target_data.append(target_data_batch_mod)
            
        self.y = np.asarray(target_data, dtype=np.uint8)
        self.x = np.asarray(input_data)
        input_padded = keras.preprocessing.sequence.pad_sequences(np.asarray(input_data), padding='post')
        target_padded = keras.preprocessing.sequence.pad_sequences(np.asarray(target_data, dtype=np.uint8), padding='post')
            
        #input_padded = np.asarray(input_data)
        #target_padded = np.asarray(target_data)
        #print(type(target_padded))
    
        #return input_padded,target_padded
        return tf.convert_to_tensor(input_padded), tf.convert_to_tensor(target_padded)

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_id))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)
    '''      
    def __data_generation(self, list_id_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.epoch_length))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, id_ in enumerate(list_id_temp):
            # Store sample
            X[i,] = np.load(self.path_main+id_+'.npy')

            # Store class
            y[i] = np.load(self.path_main+self)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    '''
    #def get_y(self):
    #    return self.y

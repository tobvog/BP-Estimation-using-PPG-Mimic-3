import numpy as np
#import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    '''! Datagenerator for big data using Keras for 2 equal outputs.'''

    def __init__(self, path_main, list_id, batch_size, n_sample=624, n_classes=2, shuffle=True):
        ##
        # @brief This constructor initalizes the DataGenerator object.
        # @param path_main      The main path which includes the preprocessed data.
        # @param list_id        The list of the subject IDs.                 
        # @param batch_size     The size of each data batch.
        # @param n_sample       The number of samples in each data instance. Default is 624.
        # @param n_classes      The number of output classes. Default is 2.
        # @param shuffle        Whether to shuffle the data after each epoch. Default is True.
        ##
        
        self.path_main = path_main
        self.list_id = list_id
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_sample = n_sample
        
        ## @brief Index of the actual ID from the parameter list_id. 
        ##
        self._id_idx = 0     
        ## @brief Total number of batches. 
        ##        
        self._nr_batches = 0 
        ## @brief Last index of periods of actual subject.
        ##
        self._last_idx = 0
        ## @brief Input data of actual subject.
        ##
        self._dev0 = np.load(self.path_main+"derivations/dev0/"+self.list_id[0], allow_pickle=True)
        ## @brief Target data of actual subject.
        ##
        self._target = np.load(self.path_main+"ground_truth/nn/"+self.list_id[self._id_idx])
        self.on_epoch_end()

    def __count_batches(self):
        ##
        # @brief    This method count the total number of batches.
        # @return   Total number of batches.
        ##
        if self._nr_batches == 0:
            for nr, sub in enumerate(self.list_id):
                print("Counting Subject no ", nr+1)
                n_epochs = len(np.load(self.path_main+"derivations/dev0/"+sub))
                self._nr_batches += int(np.ceil(n_epochs/self.batch_size))
        return self._nr_batches
        

    def __len__(self):
        ##
        # @brief This method count the total number of batches.
        # @return Total number of batches.
        ##
        return self.__count_batches()


    def __getitem__(self, idx):
        ##
        # @brief This method returns a batch of data.
        # @return A batch of data
        ##
        x, y = self.__data_generation()
        return [x[0], x[1]], y


    def on_epoch_end(self):
        ## 
        # @brief This method updates the indexes after each epoch.
        ##
        self._last_idx += self.batch_size
        if self._last_idx >= len(self._target):
            self._last_idx = 0
            self._id_idx += 1
            if self._id_idx == len(self.list_id):
                self._id_idx = 0
            self.__load_data()


    def __data_generation(self):
        ##
        # @brief This method generate one batch.
        # @return A batch of data
        ##
        x1 = np.zeros((self.batch_size, self.n_sample))
        y = np.zeros((self.batch_size, self.n_classes))
        
        for i in range(self._last_idx, len(self._target)):
        
            i0 = i-self._last_idx
            if i==len(self._target)-1:              
                if self._id_idx==len(self.list_id)-1:
                    break
                else:
                    self._last_idx = 0
                    self._id_idx += 1 
                    self.__load_data()
            
            elif i0==self.batch_size-1:
                x1[i0] = self._dev0[i]
                y[i0] = self._target[i]
                self._last_idx = i
                break
            else:
                x1[i0] = self._dev0[i]
                y[i0] = self._target[i]

        x = np.asarray([x1, x1])
        x = np.reshape(x, (2,1,self.batch_size,624))
        y = np.asarray(y)

        return x, y


    def __load_data(self):
        ##
        # @brief This method loads data of the next subject.
        ##      
        self._dev0 = np.load(self.path_main+"derivations/dev0/"+self.list_id[self._id_idx], allow_pickle=True)
        self._target = np.load(self.path_label+self.list_id[self._id_idx])
            


        
        
        
        
        
        
        
        
        
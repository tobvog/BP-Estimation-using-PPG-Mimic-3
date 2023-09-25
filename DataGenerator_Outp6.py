import numpy as np
import random
from tensorflow.keras.utils import Sequence
## @brief Datagenerator for 6 Inputs of 3 different sources. 
## @details This Datagenerator can be used for big data which would overload the RAM.  

class DataGenerator(Sequence):
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
        ## @brief First derivation of input data of actual subject.
        ##
        self._dev1 = np.load(self.path_main+"derivations/dev1/"+self.list_id[0], allow_pickle=True)
        ## @brief Second derivation of input data of actual subject.
        ##
        self._dev2 = np.load(self.path_main+"derivations/dev2/"+self.list_id[0], allow_pickle=True)
        ## @brief Target data of actual subject.
        ##
        self._target = np.load(self.path_main+"ground_truth/nn/"+self.list_id[0])
        self.on_epoch_end()

    
    def __count_batches(self):
        ##
        # @brief    This method count the total number of batches.
        # @return   Total number of batches.
        ##
        if self._nr_batches == 0:
            print("Counting Subjects")
            for nr, sub in enumerate(self.list_id):
                n_epochs = len(np.load(self.path_main+"derivations/dev0/"+sub))
                self._nr_batches += int(n_epochs/self.batch_size)
                
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
        return [x[0], x[1], x[2], x[3], x[4], x[5]], y

    def on_epoch_end(self):
        ## 
        # @brief This method updates the indexes after each epoch.
        ##
        if self.shuffle==True:
            random.shuffle(self.list_id)
        self._last_idx = 0
        self._id_idx = 0
        self.__load_data()
            

    def __data_generation(self):
        ##
        # @brief This method generate one batch.
        # @return A batch of data
        ##
        x1, x2, x3 = [np.zeros((self.batch_size, self.n_sample)) for i in range(0,3)]
        y = np.zeros((self.batch_size, self.n_classes))
         
        
        for i in range(self._last_idx, len(self._dev0)):
            if self._id_idx==len(self.list_id):
                break
                            
            i0 = i-self._last_idx    
            # last data of batch
            if i0>=self.batch_size-1:
                x1[i0] = self._dev0[i]
                x2[i0] = self._dev1[i]
                x3[i0] = self._dev2[i]
                y[i0] = self._target[i]
                self._last_idx = i
                break
            
            # fill batch
            else:
                x1[i0] = self._dev0[i]
                x2[i0] = self._dev1[i]
                x3[i0] = self._dev2[i]
                y[i0] = self._target[i]
              
        x = np.asarray([x1, x2, x3, x1, x2, x3])
        x = np.reshape(x, (6,1,self.batch_size,624))
        y = np.asarray(y)

        return x, y


    def __load_data(self):
        ##
        # @brief This method loads data of the next subject.
        ##   
        self.dev0 = np.load(self.path_main+"derivations/dev0/"+self.list_id[self._id_idx], allow_pickle=True)
        self.dev1 = np.load(self.path_main+"derivations/dev1/"+self.list_id[self._id_idx], allow_pickle=True)
        self.dev2 = np.load(self.path_main+"derivations/dev2/"+self.list_id[self._id_idx], allow_pickle=True)
        self._target = np.load(self.path_main+"ground_truth/nn/"+self.list_id[self._id_idx])
            


        
        
        
        
        
        
        
        
        
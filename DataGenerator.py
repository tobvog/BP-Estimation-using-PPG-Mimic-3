import numpy as np
import tensorflow.keras as keras
import tensorflow as tf


class DataGenerator(keras.utils.Sequence):
    def __init__(self, path_main, list_id, path_label, batch_size, n_sample=624, n_epoch=128, n_classes=2, validation_split=0.2, shuffle=True):
        self.path_main = path_main
        self.path_label = path_label
        self.list_id = list_id
        self.id_idx = 0 
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_sample = n_sample
        self.n_epoch = n_epoch
        self.last_idx = 0
        self.dev0 = np.load(self.path_main+"derivations/dev0/"+self.list_id[0], allow_pickle=True)
        self.dev1 = np.load(self.path_main+"derivations/dev1/"+self.list_id[0], allow_pickle=True)
        self.dev2 = np.load(self.path_main+"derivations/dev2/"+self.list_id[0], allow_pickle=True)
        self.target = np.load(self.path_label+self.list_id[self.id_idx])
        self.on_epoch_end()

    def __count_batches(self):
        n_batches = 0
        for sub in self.list_id:
            n_epochs = len(np.load(self.path_main+"derivations/dev0/"+sub))
            n_batches += int(np.ceil(n_epochs/self.batch_size))
        
        return n_batches
        

    def __len__(self):
        return self.__count_batches()
    
    def __getitem__(self, idx):
        #batch_id = self.list_id[idx*self.batch_size : (idx+1)*self.batch_size]

        x, y = self.__data_generation()

        return [x[0], x[1], x[2], x[3], x[4], x[5]], y


    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_id))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)


    def __data_generation(self):
        x1, x2, x3 = [np.zeros((self.n_epoch, self.n_sample)) for i in range(0,3)]
        y = np.zeros((self.n_epoch, self.n_classes))
        
        for i in range(self.last_idx, len(self.target)):
        
            i0 = i-self.last_idx
            if i==len(self.target)-1:
                #a = len(self.target)-self.last_idx
                x1[i0:] = self.dev0[i]
                x2[i0:] = self.dev1[i]
                x3[i0:] = self.dev2[i]
                y[i0:] = self.target[i]
                
                if self.id_idx==len(self.list_id)-1:
                    break
                else:
                    self.last_idx = 0
                    self.id_idx += 1 
                    self.load_data()
            
            elif i0==self.n_epoch-1:
                x1[i0] = self.dev0[i]
                x2[i0] = self.dev1[i]
                x3[i0] = self.dev2[i]
                y[i0] = self.target[i]
                self.last_idx = i
                break
            else:
                x1[i0] = self.dev0[i]
                x2[i0] = self.dev1[i]
                x3[i0] = self.dev2[i]
                y[i0] = self.target[i]
              
        
        
        x = np.asarray([x1, x2, x3, x1, x2, x3])
        x = np.reshape(x, (6,1,self.batch_size,624))
        y = np.asarray(y)

    
        return x, y


    def load_data(self):
        self.dev0 = np.load(self.path_main+"derivations/dev0/"+self.list_id[self.id_idx], allow_pickle=True)
        self.dev1 = np.load(self.path_main+"derivations/dev1/"+self.list_id[self.id_idx], allow_pickle=True)
        self.dev2 = np.load(self.path_main+"derivations/dev2/"+self.list_id[self.id_idx], allow_pickle=True)
        self.target = np.load(self.path_label+self.list_id[self.id_idx])
            


        
        
        
        
        
        
        
        
        
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
## @brief Library for the classical machine learning.  
## @details This Library provides methods for the classical machine learning part of the blood pressure estimation by Slapnicar et al.
class Classical_ML():
    @staticmethod
    def __spec_flatten(x_in=None, y_in=None, only_y=False):  
        ##
        # @brief This method flat arrays
        # @param x_in           Input feature data.
        # @param y_in           Input target data.                 
        # @param only_y         Bool value if only target data should be flatten.
        # @return               Flattened data.
        ##
        x_out, y_out = [], []
        for sub in range(0, len(y_in)):
            for cyc in range(0, len(y_in[sub])):
                if only_y==False:    
                    temp_x = x_in[sub][cyc]
                temp_y = y_in[sub][cyc]
                if np.isnan(temp_y[0])==False and np.isnan(temp_y[1])==False:
                    if only_y==False:
                        x_out.append(temp_x)
                    y_out.append(temp_y)
        if only_y==False:
            return np.array(x_out, dtype=object), np.array(y_out)
        else:
            return np.array(y_out)
        
###############################################################################
###############################################################################
###############################################################################         
    @staticmethod
    def __batch(iterable, n=1):
        ##
        # @brief This method organize batches and iterate through them.
        # @param iterable       Input data which should be iterated by batches.
        # @param n              Batch size.                
        # @return               Actual batch of the iteration.
        ##
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
      
###############################################################################
###############################################################################
###############################################################################              
    def dummy(self, y, train_index, test_index):
        ##
        # @brief This method realizes the "Dummy" training of Slapnicar BP Estimation.
        # @param y              Target data.
        # @param train_index    Indices of training data. 
        # @param test_index     Indices of test data.     
        # @return               Mean absolute error of systolic and diastolic blood pressure. 
        ##
        y_train, y_test = y[train_index], y[test_index]
        y_train = self.__spec_flatten(y_in=y_train, only_y=True)
        y_test = self.__spec_flatten(y_in=y_test, only_y=True)
        
        y_pred_sbp = np.full(len(y_test), np.mean(y_train[:,0])) 
        y_pred_dbp = np.full(len(y_test), np.mean(y_train[:,1])) 
        
        mae_sbp = mean_absolute_error(y_test[:,0], y_pred_sbp)
        mae_dbp = mean_absolute_error(y_test[:,1], y_pred_dbp)

        return mae_sbp, mae_dbp
    
###############################################################################
###############################################################################
###############################################################################      
    def rfregression(self, ids, y, train_index, test_index, path, label, batch_size=64, n_jobs=1):
        ##
        # @brief This method realizes the RandomForest training of Slapnicar BP Estimation.
        # @param ids            List of subject identification numbers.
        # @param y              Target data.
        # @param train_index    Indices of training data. 
        # @param test_index     Indices of test data.  
        # @param path           Path of all input data.
        # @param label          Set the label which should be classified. 0=Systolic blood pressure and 1=Diastolic blood pressure.
        # @param batch_size     Batch size of training. Default=64.
        # @param n_jobs         Number of RAM that calculate. Default=1.
        # @return               Mean absolute error of test data.  
        ##
        
        nr_batches = int(len(train_index)/batch_size)
        clf = RandomForestRegressor(n_estimators=13, verbose=2, n_jobs=n_jobs, warm_start=True)
        
        nr_batch = 1
        for mini_batch in self.__batch(iterable=train_index, n=batch_size):
            print("Batch nr: "+str(nr_batch)+" of "+str(nr_batches))
            nr_batch += 1
            x_train = np.array([np.load(path+"feature/"+subject, allow_pickle=True) for subject in ids[mini_batch]], dtype=object)
            y_train = y[mini_batch]
            x_train, y_train = self.__spec_flatten(x_train, y_train)
            y_train = y_train[:,label]
           
            clf.fit(x_train, y_train)
            clf.n_estimators += 13
        
        x_test = np.array([np.load(path+"feature/"+subject, allow_pickle=True) for subject in ids[test_index]], dtype=object)
        y_test = y[test_index]
        x_test, y_test = self.__spec_flatten(x_test, y_test)
        y_test = y_test[:,label]
        y_pred = clf.predict(x_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        
        return mae
    
###############################################################################
###############################################################################
###############################################################################      
    def rfregression_pers(self, ids, y, train_index, test_index, path, label, batch_size=64, n_jobs=1):
        ##
        # @brief This method realizes the RandomForest training with personalization of Slapnicar BP Estimation.
        # @param ids            List of subject identification numbers.
        # @param y              Target data.
        # @param train_index    Indices of training data. 
        # @param test_index     Indices of test data.  
        # @param path           Path of all input data.
        # @param label          Set the label which should be classified. 0=Systolic blood pressure and 1=Diastolic blood pressure.
        # @param batch_size     Batch size of training. Default=64.
        # @param n_jobs         Number of RAM that calculate. Default=1.  
        # @return               Mean absolute error of test data.  
        ##
        nr_batches = int(len(y)/batch_size)
        clf = RandomForestRegressor(n_estimators=13, warm_start=True, verbose=2, n_jobs=n_jobs)

        x_test = np.array([np.load(path+"feature/"+subject, allow_pickle=True) for subject in ids[test_index]], dtype=object)
        y_test = y[test_index]
        x_test, y_test = self.__spec_flatten(x_test, y_test)
        y_test = y_test[:,label]

        nr_cyc = int(len(x_test) * 0.2)
        x_test = x_test[nr_cyc:]
        y_test = y_test[nr_cyc:]
        nr_batch = 1
        
        for mini_batch in self.__batch(train_index, batch_size):
            print("Batch nr: "+str(nr_batch)+" of "+str(nr_batches))
            nr_batch += 1
            x_train = np.array([np.load(path+"feature/"+subject, allow_pickle=True) for subject in ids[mini_batch]], dtype=object)
            y_train = y[mini_batch]
            x_train, y_train = self.__spec_flatten(x_train, y_train)
            y_train = y_train[:,label]
            
            clf.fit(x_train, y_train)
            clf.n_estimators += 13

        clf.fit(x_test[:nr_cyc], y_test[:nr_cyc])
        y_pred = clf.predict(x_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        
        return mae
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    

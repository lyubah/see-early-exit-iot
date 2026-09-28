


import pickle
import argparse
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import copy
 
 
class LoadData:
 
    def __init__(self):
        self.data = None
        self.labels_array = None
        self.n_window = None
        self.dataset_name = None
       
    def Read(self, datasetName=None):
   
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_name', type=str, default='EMGPhysical', required=False)
        parser.add_argument('--model_type', type=str, help="model", required=False, default="classic")
        args, _ = parser.parse_known_args()  # tolerate the caller's own CLI flags
       
        if datasetName is None:
            dataset_name = args.dataset_name  
        else:
            dataset_name = datasetName
 
        self.dataset_name = dataset_name
           
        model_type = args.model_type
        file_path = rf'Datasets/{dataset_name}_dataLabels.pkl'
        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)
 
        self.data = data_dict['data']
        self.labels_array = data_dict['labels']
 
        m, n = self.data.shape[::2]                                                      
        self.n_window, n_channel, n_data = self.data.shape
 
    def _flatten_triaxial(self, X):
     
        n_windows, n_sensors, n_timestamps = X.shape
        assert n_timestamps % 3 == 0, \
            f"Expected timestamps divisible by 3 for triaxial data, got {n_timestamps}"
        T = n_timestamps // 3  # e.g. 200
 
        # Split the timestamp axis into (3_axes, T)
        # Result: (windows, n_sensors, 3_axes, T)
        X_reshaped = X.reshape(n_windows, n_sensors, 3, T)
 
        # Transpose to (windows, T, n_sensors, 3_axes)
        # so that flattening last two dims gives: s1x, s1y, s1z, s2x, s2y, s2z, ...
        X_reordered = X_reshaped.transpose(0, 3, 1, 2)
        # shape: (windows, T, n_sensors, 3_axes)
 
        # Flatten last 3 dims → (windows, T * n_sensors * 3)
        return X_reordered.reshape(n_windows, T * n_sensors * 3)
 
    def _flatten_standard(self, X):
 
        n_windows, n_channels, n_timestamps = X.shape
        return np.swapaxes(X, 1, 2).reshape(n_windows, n_channels * n_timestamps)
 
    def _flatten(self, X):
        """Route to the correct flattening strategy based on dataset name."""
        if self.dataset_name in ['Shoaib', 'PAMAP2']:
            return self._flatten_triaxial(X)
        else:
            return self._flatten_standard(X)
 
    def SplitData(self):
       
        X = self.GetDate()
        y = self.GetLabel()
 
        lst = list(range(0, self.GetWindow()))
       
        # First split: 80% (Train + Validation) and 20% (Test)
        X_temp, X_test_ind, l_temp, l_test = train_test_split(
            lst, y, test_size=0.20, random_state=42)
 
        # Second split: 60% Train and 20% Validation
        X_train_ind, X_val_ind, l_train, l_val = train_test_split(
            X_temp, l_temp, test_size=0.25, random_state=42)
 
        # data splits
        self.X_train = X[X_train_ind, :, :]
        self.X_test  = X[X_test_ind,  :, :]
        self.X_val   = X[X_val_ind,   :, :]
 
        # label splits
        self.y_train = y[X_train_ind]
        self.y_test  = y[X_test_ind]
        self.y_val   = y[X_val_ind]
 
        # flatten using the appropriate strategy
        self.train_X_flatten = self._flatten(copy.deepcopy(self.X_train))
        self.val_X_flatten   = self._flatten(copy.deepcopy(self.X_val))
        self.test_X_flatten  = self._flatten(copy.deepcopy(self.X_test))
               
    def GetDate(self):
        return self.data
   
    def GetLabel(self):
        return self.labels_array
   
    def GetWindow(self):
        return self.n_window
   
    def GetTrainX(self):
        return self.train_X_flatten
   
    def GetValX(self):
        return self.val_X_flatten
   
    def GetTestX(self):
        return self.test_X_flatten
   
    def GetYtrain(self):
        return self.y_train
   
    def GetYtest(self):
        return self.y_test
   
    def GetYval(self):
        return self.y_val
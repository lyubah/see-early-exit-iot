import pickle
import argparse
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
import copy

# seg_layout detects + verifies the SEG_SIZE-axis layout of triaxial data (instead of
# assuming it), so every model flattens triaxial windows the same way.
import seg_layout


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
        args = parser.parse_args()
        
        if datasetName is None:
            dataset_name = args.dataset_name   
        else:
            dataset_name = datasetName

        self.dataset_name = dataset_name
            
        model_type = args.model_type
        # Datasets/ at the current directory, else the repo-level Datasets/ folder.
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _candidates = [os.path.join('Datasets', f'{dataset_name}_dataLabels.pkl'),
                       os.path.join(_repo_root, 'Datasets', f'{dataset_name}_dataLabels.pkl')]
        file_path = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)

        self.data = data_dict['data']
        self.labels_array = data_dict['labels']

        m, n = self.data.shape[::2]                                                      
        self.n_window, n_channel, n_data = self.data.shape

    def _flatten(self, X):
        """Flatten a raw (windows, sensors, SEG_SIZE) split to (windows, features),
        time-major with the 3 axes grouped per timestep (s1x, s1y, s1z, s2x, ...).

        Routes through seg_layout, which DETECTS + verifies the SEG_SIZE-axis layout
        from the data (axis_major / interleaved / time) instead of hard-coding the
        old `Shoaib/PAMAP2 -> reshape(sensors,3,T)` assumption. If the data really is
        axis-major this is bit-identical to the previous _flatten_triaxial; otherwise
        it corrects the previously-leaking ordering. Non-triaxial datasets are
        unchanged (== the old swapaxes-based _flatten_standard)."""
        return seg_layout.flatten_full(X, self.dataset_name)

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

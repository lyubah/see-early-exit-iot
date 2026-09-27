import pickle
from sklearn.model_selection import train_test_split
import os
import sys
# SEE-vRF forest with exits (see_rf/vrf/rf_with_exits). Kept importable as top-level
# `RandomForest` so the pickled model loads the same way in Training_Inference.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vrf", "rf_with_exits"))
from RandomForest import RandomForest
import numpy as np
import argparse
from DataLoad import LoadData


def Train_and_Save(dataset_name, n_estimators, tree_depth, tree_splits, proportions):
    file_path = f'Datasets/{dataset_name}_dataLabels.pkl'
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)

    data = data_dict['data']          # data
    labels_array = data_dict['labels'] # labels

    n_window, n_channel, n_data = data.shape  # data shape
    lst = np.arange(n_window)


    loader = LoadData()
    loader.Read(dataset_name)
    loader.SplitData()

    X_train = loader.GetTrainX()
    X_val   = loader.GetValX()
    X_test  = loader.GetTestX()

    y_train = loader.GetYtrain()
    y_val   = loader.GetYval()
    y_test  = loader.GetYtest()

    '''X_train_len = len(y_train)
    X_val_len   = len(y_val)
    X_test_len  = len(y_test)'''

    '''# First split: 80% (Train + Validation) and 20% (Test)
    X_temp, X_test_ind, l_temp, l_test = train_test_split(lst, labels_array, test_size=0.20, random_state=42)
    X_train_ind, X_val_ind, l_train, l_val = train_test_split(X_temp, l_temp, test_size=0.25, random_state=42)  # 0.25 * 80% = 20%
'''

    '''# data
    X_train = data[X_train_ind].reshape(len(X_train_ind), n_channel*n_data)
    X_test =  data[X_test_ind].reshape(len(X_test_ind), n_channel*n_data)
    #X_val = data[X_val_ind].reshape(len(X_val_ind), n_channel*n_data)

    # labels
    y_train = l_train
    y_test = l_test'''

    nb_clss = len(np.unique(y_train))

    clf = RandomForest(n_trees=n_estimators, max_depth= tree_depth)

    # Training
    clf.fit(X_train, y_train, proportions, tree_splits)

    # Shut down the executor
    if hasattr(clf, "executor") and clf.executor:
         clf.executor.shutdown(wait=True)
         clf.executor = None

    # Save the model
    with open(f"{dataset_name}_trained_model.pkl", "wb") as f:
        pickle.dump(clf,f)

    # Save the X_train and y_train and nb_classes
    np.save(f"{dataset_name}_test_data.npy", X_test)
    np.save(f"{dataset_name}_test_labels.npy", y_test) 
    np.save(f"{dataset_name}_nb_classes.npy", np.array([nb_clss]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "RF-V Training")
    parser.add_argument("--dataset_name", type=str, default= "Epilepsy", help = "The Dataset name")
    parser.add_argument("--n_est", type=int, default=83, help = "The number of estimators")
    parser.add_argument("--max_depth", type=int, default=17, help = "The max depth")
    parser.add_argument("--tree_splits", type=float, help = "Tree splits", nargs="+", default=[0.5, 1])
    parser.add_argument("--proportions", type=float, help = "Data proportions", nargs="+", default=[0.25, 1])
    args = parser.parse_args()
    Train_and_Save(args.dataset_name, args.n_est, args.max_depth, args.tree_splits, args.proportions)
  


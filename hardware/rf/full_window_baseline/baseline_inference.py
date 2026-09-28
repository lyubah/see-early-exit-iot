from sklearn.model_selection import train_test_split
import numpy as np
import time
import pickle
#from RandomForest import RandomForest
import concurrent.futures
 
 
def full_window_time_sec(window_len, fs_base):
    return float(window_len) / float(fs_base)
 
 
def stage_acquisition_times(split_points, window_len, fs_base):
 
    T_window = full_window_time_sec(window_len, fs_base)
    prev = 0.0
    out = []
    for p in split_points:
        p = float(p)
        seg_prop = max(0.0, p - prev)
        out.append(seg_prop * T_window)
        prev = p
    return out
 
def entropy_f(probabilities):   # probabilities is of shape (n_windows, n_classes)
    epsilon = 1e-5  # to avoid taking the logarithm of zero
    return -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
 
 
def read_configuration_return_results(window_len, fs_base, dataset_name, sensor_on): # proportions here are the data%
 
    model_file = f'saved_pkl/{dataset_name}_trained_model.pkl'
    X_test_file = f'saved_pkl/{dataset_name}_test_data.npy'
    y_test_file = f'saved_pkl/{dataset_name}_test_labels.npy'
    num_classes_file = f'saved_pkl/{dataset_name}_nb_classes.npy'
 
   
 
    # load the model
    with open(model_file, 'rb') as f1:
         clf = pickle.load(f1)
   
 
    # load data and labels
    X_test = np.load(X_test_file)
    y_test = np.load(y_test_file)
    nb_clss = int(np.load(num_classes_file)[0])
 
    T_window = full_window_time_sec(window_len, fs_base) # **********************************************
 
    list_of_inference_results = []
   
    for w in range(min(100, len(X_test))):
 
        sensor_on(verbose=True)
        time.sleep(T_window)
   
        t1 = time.time()
        pred = clf.predict(X_test[w].reshape(1, -1))
        t2 = time.time()
       
       
        list_of_inference_results.append({'t_start': t1, 't_end': t2, 'total': t2-t1, 'true_label': y_test[w], 'prediction': pred[0], 'correctness': y_test[w]==pred[0]})
       
               
               
    return list_of_inference_results

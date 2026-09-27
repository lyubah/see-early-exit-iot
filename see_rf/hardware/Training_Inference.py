from sklearn.model_selection import train_test_split
import numpy as np
import time
import pickle
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vrf", "rf_with_exits"))
from RandomForest import RandomForest  # needed so the pickled forest can be loaded
import concurrent.futures




def entropy_f(probabilities):   # probabilities is of shape (n_windows, n_classes)
    epsilon = 1e-5  # to avoid taking the logarithm of zero
    return -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)


def read_configuration_return_results(dataset_name, num_exits, proportions, th_list): # proportions here are the data%

    model_file = f'{dataset_name}_trained_model.pkl'
    X_test_file = f'{dataset_name}_test_data.npy'
    y_test_file = f'{dataset_name}_test_labels.npy'
    num_classes_file = f'{dataset_name}_nb_classes.npy'

    # load the model
    with open(model_file, 'rb') as f1:
         clf = pickle.load(f1)
    

    # load data and labels
    X_test = np.load(X_test_file)
    y_test = np.load(y_test_file)
    nb_clss = int(np.load(num_classes_file)[0])

    list_of_inference_results = []
    
    # Activate the executor 
    if hasattr(clf, "executor") and clf.executor is None:
        clf.executor = concurrent.futures.ProcessPoolExecutor(max_workers=5)


    for w in range(len(X_test)):

        previous_exit_time_1 = 0
        previous_exit_time_2 = 0
        previous_exit_time_3 = 0
        not_classified = True

        # check first exit
        exit_level = 1
        num_samples = int(X_test.shape[1] * proportions[0])
        subset = X_test[w,:num_samples].reshape(1,-1)
        y_subset = y_test[w]
        exit_level = 1
        start_nodes = None

        t1 = time.time()
        predictions, exit_nodes, probabilities = clf.predict(subset, nb_clss, exit_level, start_nodes=start_nodes)
        entropy = entropy_f(probabilities)
        t2 = time.time() # exiting the first exit
        t_start = t1; e1 = t2; e2=-1; e3=-1; e4=-1; total = t2-t1
        
        if entropy[0] <= th_list[0]:
            # store results
            list_of_inference_results.append({'t_start': t_start,'t1': e1, 't2': e2, 't3': e3, 't4': e4, 'total': total, 'true_label': y_subset, 'prediction': predictions[0], 'correctness': y_subset== predictions[0], 'exit_taken': 1, 'data%': proportions[0]})
            not_classified = False
        else:
            previous_exit_time_1 = t2
            previous_total = total
            
        i = 1
      
        while (not_classified) and (i+1 < num_exits):
            exit_level = i+1  
            start_nodes_2 = exit_nodes
            num_samples = int(X_test.shape[1] * proportions[i])
            subset = X_test[w,:num_samples].reshape(1,-1)

            t1 = time.time()
            predictions, exit_nodes, prob = clf.predict(subset, nb_clss, exit_level, start_nodes=start_nodes_2)
            entropy = entropy_f(prob)
            t2 = time.time()

           
            if exit_level == 2:
                e1 = previous_exit_time_1; e2 = t2; e3 = -1; e4 = -1; total = previous_total + t2-t1
                previous_exit_time_2 = t2
                previous_total = total

            else: # exit 3
                e1 = previous_exit_time_1; e2 = previous_exit_time_2; e3 = t2; e4 = -1; total = previous_total + t2-t1
                previous_exit_time_3 = t2
                previous_total = total
            
            if entropy[0] <= th_list[i]:
                list_of_inference_results.append({'t_start': t_start,'t1': e1, 't2': e2, 't3': e3, 't4': e4, 'total': total, 'true_label': y_subset, 'prediction': predictions[0], 'correctness': y_subset== predictions[0], 'exit_taken': exit_level, 'data%': proportions[exit_level-1]})
                not_classified = False
            else:
                i += 1

        if not_classified  and i+1 == num_exits: # we had to go through all the levels up to the last exit 
            exit_level = i+1  
            start_nodes_2 = exit_nodes
            num_samples = int(X_test.shape[1] * proportions[i])
            subset = X_test[w,:num_samples].reshape(1,-1)
        
            t1 = time.time()
            predictions, exit_nodes, _ = clf.predict(subset, nb_clss, exit_level, start_nodes=start_nodes_2)
            t2 = time.time()
            if exit_level == 2:
                e1 = previous_exit_time_1; e2 = t2; e3 = -1; e4 = -1; total = previous_total + t2-t1

            elif exit_level == 3: 
                e1 = previous_exit_time_1; e2 = previous_exit_time_2; e3 = t2; e4 = -1; total = previous_total + t2-t1
        
            else: 
                e1 = previous_exit_time_1; e2 = previous_exit_time_2; e3 = previous_exit_time_3; e4 = t2; total = previous_total + t2-t1
    
            list_of_inference_results.append({'t_start': t_start,'t1': e1, 't2': e2, 't3': e3, 't4': e4, 'total': total, 'true_label': y_subset, 'prediction': predictions[0], 'correctness': y_subset== predictions[0], 'exit_taken': exit_level, 'data%': proportions[exit_level-1]})
        
                    
                
    return list_of_inference_results
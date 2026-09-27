import pandas as pd
import numpy as np
import argparse
import sys
import copy
import pickle
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
import torch
import torch.nn.functional as F
import math


def error(y_true, y_pred): # will return a list that indicates whether each data was well classified or not
    err=(y_true == y_pred).astype(int).tolist()
    return np.array(err)

def accuracy(y_true, y_pred):
    if len(y_true)>0:
      accuracy = np.sum(y_true == y_pred) / len(y_true)
    else:
      accuracy = np.nan
    return accuracy



def entropy(probabilities):   # probabilities is of shape (n_windows, n_classes)
    epsilon = 1e-5  # to avoid taking the logarithm of zero
    return -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)


def cross_entropy(probabilities, y_true): # probabilities is of shape (n_windows, n_classes), y_true is of size (n_windows,)
  prob_pred = probabilities[np.arange(len(y_true)), y_true]
  loss = -np.log(prob_pred  + 1e-5)
  return loss # list of cross entropy


def _tree_splits(n_exits, n_tree_splits): # returns a list of n_tree_splits lists; each list corresponds to a tree split with n_exits

  l=[]
  for _ in range(n_tree_splits):
    unique_prop = set()
    a = round(np.random.uniform(0.5, 0.6),2)
    unique_prop.add(a)

    while len(unique_prop) < n_exits-1:
      a = round(np.random.uniform(a+0.2, 0.8),2)
      unique_prop.add(a)

    sort = sorted(unique_prop)
    sort.append(1)
    l.append(sort)
  return l



def run_and_record_experiment(config): # config: dictionnary with one configuration -> returns a dictionnary (that needs to be appended later on
                                                                                                 # as a row in the csv file (one single experiment))

    # the configuration parameters
    dataset_name = config['Dataset']
    num_exits = config['Num_exits']
    max_depth = config['Tree_depth']
    tree_splits = config['Tree_splits']
    proportions = config['Proportions']
    all_config_th = config['th'] # list of lists 
    #k = config['k']

    ############################ if accuracy < 50 stop ####################################
    file_path = f'Datasets/{dataset_name}_dataLabels.pkl'
    with open(file_path, 'rb') as file:
      data_dict = pickle.load(file)
    data = data_dict['data']          # data
    labels_array = data_dict['labels'] # labels
    #file_path = f'Datasets/{dataset_name}_specs.pkl'
    #with open(file_path, 'rb') as file:
       # dataSpecs = pickle.load(file)
    #SEG_SIZE = dataSpecs['SEG_SIZE']
    #CHANNEL_NB = dataSpecs['CHANNEL_NB']
    #CLASS_NB = dataSpecs['CLASS_NB']
    n_window, n_channel, n_data = data.shape  # data shape
    unique_numbers_set = set(labels_array)
    num_activities = len(unique_numbers_set)  # unique labels

    lst = list(range(0, n_window))
    total_num_data = n_window


    # First split: 80% (Train + Validation) and 20% (Test)
    X_temp, X_test_ind, l_temp, l_test = train_test_split(lst, labels_array, test_size=0.20, random_state=0)

    # Second split: Split 80% into 60% (Train) and 20% (Validation)
    X_train_ind, X_val_ind, l_train, l_val = train_test_split(X_temp, l_temp, test_size=0.25, random_state=0)  # 0.25 * 80% = 20%


    # data
    X_train = data[X_train_ind,:,:]
    X_test =  data[X_test_ind,:,:]
    X_val = data[X_val_ind,:,:]

    # labels
    y_train = labels_array[X_train_ind]
    y_test = labels_array[X_test_ind]
    y_val = labels_array[X_val_ind]


    classes = np.unique(y_train)
    #print(f'We are using the {dataset_name} dataset')
    #print("the classes used in this dataset: ", classes)
    n_classes = len(classes)

    # flatten the data
    input_data_RF_train = copy.deepcopy(X_train)
    train_flatten = input_data_RF_train.reshape(len(y_train), n_channel*n_data)

    input_data_RF_val = copy.deepcopy(X_val)
    val_flatten = input_data_RF_val.reshape(len(y_val), n_channel*n_data)

    input_data_RF_test = copy.deepcopy(X_test)
    test_flatten = input_data_RF_test.reshape(len(y_test), n_channel*n_data)

    #print("the dataset size: ", n_window)
    #print("the number of features after flattening the data: ", train_flatten.shape[1])


    dataset=[]
    num_exits_col = []
    depth_col = []
    tree_splits_col = [] # will contain a list
    prop = [] # will contain a list

    T_acc = []  ################# make it one list and then handle the csv file content ***********
    V_acc = []  ################# make it one list and then handle the csv file content  ***********
    Test_acc = []  ################# make it one list and then handle the csv file content  ***********

    #E_TH_col = []   ################################# then populate it to all exits THs

    #E_TH1_col =[]  # *************
    #E_TH2_col = [] # *************
    #E_TH3_col = [] # *************
    #E_TH4_col = [] # *************



    dataset.append(dataset_name)
    num_exits_col.append(num_exits)
    depth_col.append(max_depth)
    tree_splits_col.append(tree_splits)
    prop.append(proportions)



    #print("#################################### NEW CONFIGURATION ######################################")
    #print(f'The features portions are: {proportions}. We will use the max depth: {max_depth} and the following tree splits: {tree_splits}; we will have {num_exits} exits.')
    #print(f"the training dataset size is: ", len(X_train))
    #print(f"the validation dataset size is: ", len(X_val))
    #print(f"the test dataset size is: ", len(X_test))

    
    # train the RF
    #print("training the model")
    clf = RandomForest(n_trees=95, max_depth=max_depth)
    clf.fit(train_flatten, y_train, proportions, tree_splits)

    # check exits accuracies for training dataset (by forcing the whole training dataset at each exit level)
    #print("check exits accuracies for training dataset by forcing the whole training dataset at each exit level")

    # first exit
    num_samples_train = int(train_flatten.shape[1] * proportions[0])
    train_subset = train_flatten[:,:num_samples_train]
    exit_level = 1
    start_nodes = None
    predictions, exit_nodes, prob = clf.predict(train_subset, n_classes, exit_level, start_nodes=None)
    Train_exit_acc = float(accuracy(y_train, predictions))
    T_acc.append(Train_exit_acc)

    # next exits
    for i in range(1,num_exits):
      num_samples_train = int(train_flatten.shape[1] * proportions[i])
      train_subset = train_flatten[:,:num_samples_train]
      exit_level = i+1
      predictions, exit_nodes, prob = clf.predict(train_subset, n_classes, exit_level, start_nodes=exit_nodes)

      Train_exit_acc = float(accuracy(y_train, predictions))
      T_acc.append(Train_exit_acc)

    # now T_acc is ready
    #print("training accuracies: ", T_acc)

    # check exits accuracies for validation dataset (by forcing the whole validation dataset at each exit level)
    #print("check exits accuracies for validation dataset by forcing the whole validation dataset at each exit level")

    # first exit
    num_samples_val = int(val_flatten.shape[1] * proportions[0])
    val_subset = val_flatten[:,:num_samples_val]
    exit_level = 1
    start_nodes = None
    predictions, exit_nodes, prob = clf.predict(val_subset, n_classes, exit_level, start_nodes=None)

    Val_exit_acc = float(accuracy(y_val, predictions))
    V_acc.append(Val_exit_acc)

    for i in range(1,num_exits):
      num_samples_val = int(val_flatten.shape[1] * proportions[i])
      val_subset = val_flatten[:,:num_samples_val]
      exit_level = i+1
      predictions, exit_nodes, prob = clf.predict(val_subset, n_classes, exit_level, start_nodes=exit_nodes)

      Val_exit_acc = float(accuracy(y_val, predictions))
      V_acc.append(Val_exit_acc)

    # now V_acc is ready
    #print("validation accuracies: ", V_acc)

    # check exits accuracies for testing dataset (by forcing the whole testing dataset at each exit level)
    #print("check exits accuracies for testing dataset by forcing the whole testing dataset at each exit level")

    # first exit
    num_samples_test = int(test_flatten.shape[1] * proportions[0])
    test_subset = test_flatten[:,:num_samples_test]
    exit_level = 1
    start_nodes = None
    predictions, exit_nodes, prob = clf.predict(test_subset, n_classes, exit_level, start_nodes=None)

    Test_exit_acc = float(accuracy(y_test, predictions))
    Test_acc.append(Test_exit_acc)

    # next exits
    for i in range(1,num_exits):
      num_samples_test = int(test_flatten.shape[1] * proportions[i])
      test_subset = test_flatten[:,:num_samples_test]
      exit_level = i+1
      predictions, exit_nodes, prob = clf.predict(test_subset, n_classes, exit_level, start_nodes=exit_nodes)

      Test_exit_acc = float(accuracy(y_test, predictions))
      Test_acc.append(Test_exit_acc)

    # now Test_acc is ready
    #print("Test accuracies: ", Test_acc)

    ########################################################################################################################################################
    while len(T_acc)<4:
      T_acc.append(-1)

    while len(V_acc)<4:
      V_acc.append(-1)

    while len(Test_acc)<4:
      Test_acc.append(-1)
    ##########################################################################################################################################################

    #print('Let us try to exit these levels whenever we are confident enough!')

    # evaluate train dataset with exits ; check how many data exited through every each exit and calculate the exits accuracies and the averaged accuracy
    #print('Let us do that for the training dataset')
    # predict using proportions[0] of the features and exit at level 1 if possible




    list_of_rows = []
    for thh in range(len(all_config_th)):

      T_acc_exit = []  ################# make it one list and then handle the csv file content **************
      V_acc_exit = []  ################# make it one list and then handle the csv file content ****************
      Test_acc_exit = []  ################# make it one list and then handle the csv file content ***************

      T_perc_taken = [] ################# make it one list and then handle the csv file content ******************
      V_perc_taken = []  ################# make it one list and then handle the csv file content ******************
      Test_perc_taken = []  ################# make it one list and then handle the csv file content *****************

      Total_train_accuracy = [] # average with weights  **************************
      Total_validation_accuracy = [] # average with weights ********************
      Total_test_accuracy = [] # average with weights **********************

      entire_data_perc_taken = [] ################# make it one list and then handle the csv file content *******************


      

      th_combination = all_config_th[thh] # one single thresholds list that corresponds for onr single configuration
      threshold = round(th_combination[0],2) #****************************************
      avg1 = 0
      num_samples = int(train_flatten.shape[1] * proportions[0])
      train_subset = train_flatten[:,:num_samples]
      train_flatten_1 = train_flatten
      y_train_subset_1 = y_train
      exit_level = 1
      start_nodes = None
      #print("the number of data for level 1: ", len(train_subset))
      #print(f"we are using {int(proportions[0]*100)}% of the features")
      predictions, exit_nodes, prob_1 = clf.predict(train_subset, n_classes, exit_level, start_nodes=None)


      # check entropy
      entropy_1 = entropy(prob_1) # a list of entropies

      # check if one of the inputs were classifed wrongly (in terms of confidence)
      indices = np.argwhere(entropy_1 > threshold).flatten()
      #E_TH_col.append(round(th_combination[0],2))
      #print("********************************** E_TH1: ", E_TH1)

      good_confidence_indices = np.argwhere(entropy_1 <= threshold).flatten() # indices of the data that exited
      acc = accuracy(y_train_subset_1[good_confidence_indices], predictions[good_confidence_indices]) # accuracy at first exit
      T_acc_exit.append(float(acc)) # append the accuracy for first exit
      T_perc_taken.append(len(good_confidence_indices)/len(train_subset)) # append the percentage of exited data for first exit
      #print("the percentage of the exited data is ", len(good_confidence_indices)/len(train_subset))
      #print("the accuracy of this exit is: ", float(acc))
      avg1 += (len(good_confidence_indices)/len(train_subset)) * (0.0 if math.isnan(acc) else float(acc)) ###############################

      # if yes, then we need to add more features to the wrongly classified samples, and start another prediction process starting from the previous exit nodes
      i=1 # proportions counter
      #th_count = 0   # to adjust the entropy threshold at each intermediate exit level
      #t = E_TH1 + k

      while (len(indices) > 0 and i+1 < num_exits):
        #print(f"Some of the input data was wrongly classified, let's move to the next level {i+1} where we use {int(proportions[i]*100)}% of the features")
        # process the rest of the data using the next level
        exit_level = i+1  # now we will exit using the next level
        start_nodes_2 = exit_nodes[indices] # the starting nodes
        num_samples = int(train_flatten.shape[1] * proportions[i])
        #print(f"the number of features used in level {i+1}: ", num_samples)
        train_flatten_2 = train_flatten_1[indices]
        train_subset_2 = train_flatten_1[indices,:num_samples]
        y_train_subset_2 = y_train_subset_1[indices]
        #print(f"the data size for level {i+1}: ", len(train_subset_2))
        predictions_2, exit_nodes_2, prob_2 = clf.predict(train_subset_2, n_classes, exit_level, start_nodes=start_nodes_2)


        # check entropy
        entropy_2 = entropy(prob_2) # a list of entropies

        # check if one of the inputs were classifed wrongly (in terms of confidence)
        indices_2 = np.argwhere(entropy_2 > round(th_combination[i],2)).flatten()
        #E_TH_col.append(round(th_combination[i],2))
        #print(f"************************************************************************* E_TH{i+1}: ", round(t+th_count*0.1,1))


        good_confidence_indices = np.argwhere(entropy_2 <= round(th_combination[i],2)).flatten()
        acc = accuracy(y_train_subset_2[good_confidence_indices], predictions_2[good_confidence_indices])
        T_acc_exit.append(float(acc))
        T_perc_taken.append(len(good_confidence_indices)/len(train_subset))
        #print("the percentage of the exited data is ", len(good_confidence_indices)/len(train_subset))
        #print("the accuracy of this exit is: ", float(acc))
        #print("we are done checking this level")
        avg1 += (len(good_confidence_indices)/len(train_subset)) * (0.0 if math.isnan(acc) else float(acc)) ###############################

        indices = indices_2
        exit_nodes = exit_nodes_2
        train_flatten_1 = train_flatten_2
        y_train_subset_1 = y_train_subset_2
        i += 1 # move to next proportion
      

      if len(indices) > 0 and i+1 == num_exits: # if we have to go through all the levels up to the last exit because some of the data is still classified wrongly
        #print(f"Some of the input data was wrongly classified, let's move to the next level {i+1} where we use {int(proportions[i]*100)}% of the features")
        # process the rest of the data using the next level
        exit_level = i+1  # now we will exit using the next level
        start_nodes_2 = exit_nodes[indices] # the starting nodes
        num_samples = int(train_flatten.shape[1] * proportions[i]) # use 60% of the features
        #print(f"the number of features used in level {i+1}: ", num_samples)
        train_flatten_2 = train_flatten_1[indices]
        train_subset_2 = train_flatten_1[indices,:num_samples]
        y_train_subset_2 = y_train_subset_1[indices]
        #print(f"the data size for level {i+1}: ", len(train_subset_2))
        predictions_2, exit_nodes_2, prob_2 = clf.predict(train_subset_2, n_classes, exit_level, start_nodes=start_nodes_2)


        # check entropy
        entropy_2 = entropy(prob_2) # a list of entropies
        #E_TH_col.append(round(th_combination[i],2))
        #print(f"***************************************************************** E_TH{i+1}: ", round(t+th_count*0.1,1))
        # check accuracy
        acc = accuracy(y_train_subset_2, predictions_2)
        T_acc_exit.append(float(acc))
        T_perc_taken.append(len(predictions_2)/len(train_subset))
        #print("the percentage of the exited data is ", len(predictions_2)/len(train_subset))
        #print("the accuracy of this exit is: ", float(acc))
        #print("we are done checking this level")
        avg1 += (len(predictions_2)/len(train_subset)) * (0.0 if math.isnan(acc) else float(acc)) ###############################

      Total_train_accuracy.append(avg1)
      # now T_acc_exit and T_perc_taken and Total_train_accuracy are ready
      #print("exits training accuracies: ", T_acc_exit)
      #print("percentages of exited data at each exit level: ", T_perc_taken)
      #print("total averaged training accuracy: ", avg1)

      #####################################################################################################################

      # evaluate validation dataset with exits ; check exits entropies; how many data exited through every each exit and calculate the averaged accuracy
      #print('Let us do that for the validation dataset')

      # predict using proportions[0] of the features and exit at level 1 if possible
      avg2 = 0
      num_samples = int(val_flatten.shape[1] * proportions[0])
      val_subset = val_flatten[:,:num_samples]
      val_flatten_1 = val_flatten
      y_val_subset_1 = y_val
      exit_level = 1
      start_nodes = None
      #print("the number of data for level 1: ", len(val_subset))
      #print(f"we are using {int(proportions[0]*100)}% of the features")
      predictions, exit_nodes, prob_1 = clf.predict(val_subset, n_classes, exit_level, start_nodes=None)

      #print("the shape of predictions: ", predictions.shape)
      #print("the shape of exit node lists: ", exit_nodes.shape)
      #print("the shape of probabilities: ", prob_1.shape)

      # check entropy
      entropy_1 = entropy(prob_1) # a list of entropies

      # check if one of the inputs were classifed wrongly (in terms of confidence)
      indices = np.argwhere(entropy_1 > threshold).flatten()


      good_confidence_indices = np.argwhere(entropy_1 <= threshold).flatten() # indices of the data that exited
      acc = accuracy(y_val_subset_1[good_confidence_indices], predictions[good_confidence_indices]) # accuracy at first exit
      V_acc_exit.append(float(acc)) # append the accuracy for first exit
      V_perc_taken.append(len(good_confidence_indices)/len(val_subset)) # append the percentage of exited data for first exit
      #print("the percentage of the exited data is ", len(good_confidence_indices)/len(val_subset))
      #print("the accuracy of this exit is: ", float(acc))
      avg2 += (len(good_confidence_indices)/len(val_subset)) * (0.0 if math.isnan(acc) else float(acc)) ###############################

      # if yes, then we need to add more features to the wrongly classified samples, and start another prediction process starting from the previous exit nodes
      i=1 # proportions counter
      

      while (len(indices) > 0 and i+1 < num_exits):
        #print(f"Some of the input data was wrongly classified, let's move to the next level {i+1} where we use {int(proportions[i]*100)}% of the features")
        # process the rest of the data using the next level
        exit_level = i+1  # now we will exit using the next level
        start_nodes_2 = exit_nodes[indices] # the starting nodes
        num_samples = int(val_flatten.shape[1] * proportions[i])
        #print(f"the number of features used in level {i+1}: ", num_samples)
        val_flatten_2 = val_flatten_1[indices]
        val_subset_2 = val_flatten_1[indices,:num_samples]
        y_val_subset_2 = y_val_subset_1[indices]
        #print(f"the data size for level {i+1}: ", len(val_subset_2))
        predictions_2, exit_nodes_2, prob_2 = clf.predict(val_subset_2, n_classes, exit_level, start_nodes=start_nodes_2)


        # check entropy
        entropy_2 = entropy(prob_2) # a list of entropies

        # check if one of the inputs were classifed wrongly (in terms of confidence)
        indices_2 = np.argwhere(entropy_2 > round(th_combination[i],2)).flatten()


        good_confidence_indices = np.argwhere(entropy_2 <= round(th_combination[i],2)).flatten()
        acc = accuracy(y_val_subset_2[good_confidence_indices], predictions_2[good_confidence_indices])
        V_acc_exit.append(float(acc))
        V_perc_taken.append(len(good_confidence_indices)/len(val_subset))
        #print("the percentage of the exited data is ", len(good_confidence_indices)/len(val_subset))
        #print("the accuracy of this exit is: ", float(acc))
        #print("we are done checking this level")
        avg2 += (len(good_confidence_indices)/len(val_subset)) * (0.0 if math.isnan(acc) else float(acc)) ###############################

        indices = indices_2
        exit_nodes = exit_nodes_2
        val_flatten_1 = val_flatten_2
        y_val_subset_1 = y_val_subset_2
        i += 1 # move to next proportion
        

      if len(indices) > 0 and i+1 == num_exits: # if we have to go through all the levels up to the last exit because some of the data is still classified wrongly
        #print(f"Some of the input data was wrongly classified, let's move to the next level {i+1} where we use {int(proportions[i]*100)}% of the features")
        # process the rest of the data using the next level
        exit_level = i+1  # now we will exit using the next level
        start_nodes_2 = exit_nodes[indices] # the starting nodes
        num_samples = int(val_flatten.shape[1] * proportions[i]) # use 60% of the features
        #print(f"the number of features used in level {i+1}: ", num_samples)
        val_flatten_2 = val_flatten_1[indices]
        val_subset_2 = val_flatten_1[indices,:num_samples]
        y_val_subset_2 = y_val_subset_1[indices]
        #print(f"the data size for level {i+1}: ", len(val_subset_2))
        predictions_2, exit_nodes_2, prob_2 = clf.predict(val_subset_2, n_classes, exit_level, start_nodes=start_nodes_2)


        # check entropy
        entropy_2 = entropy(prob_2) # a list of entropies
        # check accuracy
        acc = accuracy(y_val_subset_2, predictions_2)
        V_acc_exit.append(float(acc))
        V_perc_taken.append(len(predictions_2)/len(val_subset))
        #print("the percentage of the exited data is ", len(predictions_2)/len(val_subset))
        #print("the accuracy of this exit is: ", float(acc))
        #print("we are done checking this level")
        avg2 += (len(predictions_2)/len(val_subset)) * (0.0 if math.isnan(acc) else float(acc)) ###############################

      Total_validation_accuracy.append(avg2)
      # now V_acc_exit and V_perc_taken and Total_validation_accuracy are ready
      #print("exits validation accuracies: ", V_acc_exit)
      #print("percentages of exited data at each exit level: ", V_perc_taken)
      #print("total averaged validation accuracy: ", avg2)

      ###############################################################################################################################################################

      # evaluate test dataset with exits ; check exits entropies; how many data exited through every each exit and calculate the averaged accuracy
      #print('Let us do that for the test dataset')
      # predict using proportions[0] of the features and exit at level 1 if possible
      avg3 = 0
      num_samples = int(test_flatten.shape[1] * proportions[0])
      test_subset = test_flatten[:,:num_samples]
      test_flatten_1 = test_flatten
      y_test_subset_1 = y_test
      exit_level = 1
      start_nodes = None
      #print("the number of data for level 1: ", len(test_subset))
      #print(f"we are using {int(proportions[0]*100)}% of the features")
      predictions, exit_nodes, prob_1 = clf.predict(test_subset, n_classes, exit_level, start_nodes=None)

      #print("the shape of predictions: ", predictions.shape)
      #print("the shape of exit node lists: ", exit_nodes.shape)
      #print("the shape of probabilities: ", prob_1.shape)

      # check entropy
      entropy_1 = entropy(prob_1) # a list of entropies

      # check if one of the inputs were classifed wrongly (in terms of confidence)
      indices = np.argwhere(entropy_1 > threshold).flatten()


      good_confidence_indices = np.argwhere(entropy_1 <= threshold).flatten() # indices of the data that exited
      acc = accuracy(y_test_subset_1[good_confidence_indices], predictions[good_confidence_indices]) # accuracy at first exit
      Test_acc_exit.append(float(acc)) # append the accuracy for first exit
      Test_perc_taken.append(len(good_confidence_indices)/len(test_subset)) # append the percentage of exited data for first exit
      #print("the percentage of the exited data is ", len(good_confidence_indices)/len(test_subset))
      #print("the accuracy of this exit is: ", float(acc))
      avg3 += (len(good_confidence_indices)/len(test_subset)) * (0.0 if math.isnan(acc) else float(acc)) ###############################

      # if yes, then we need to add more features to the wrongly classified samples, and start another prediction process starting from the previous exit nodes
      i=1 # proportions counter
      th_count = 0   # to adjust the entropy threshold at each intermediate exit level
      

      while (len(indices) > 0 and i+1 < num_exits):
        #print(f"Some of the input data was wrongly classified, let's move to the next level {i+1} where we use {int(proportions[i]*100)}% of the features")
        # process the rest of the data using the next level
        exit_level = i+1  # now we will exit using the next level
        start_nodes_2 = exit_nodes[indices] # the starting nodes
        num_samples = int(test_flatten.shape[1] * proportions[i])
        #print(f"the number of features used in level {i+1}: ", num_samples)
        test_flatten_2 = test_flatten_1[indices]
        test_subset_2 = test_flatten_1[indices,:num_samples]
        y_test_subset_2 = y_test_subset_1[indices]
        #print(f"the data size for level {i+1}: ", len(test_subset_2))
        predictions_2, exit_nodes_2, prob_2 = clf.predict(test_subset_2, n_classes, exit_level, start_nodes=start_nodes_2)


        # check entropy
        entropy_2 = entropy(prob_2) # a list of entropies

        # check if one of the inputs were classifed wrongly (in terms of confidence)
        indices_2 = np.argwhere(entropy_2 > round(th_combination[i],2)).flatten()


        good_confidence_indices = np.argwhere(entropy_2 <= round(th_combination[i],2)).flatten()
        acc = accuracy(y_test_subset_2[good_confidence_indices], predictions_2[good_confidence_indices])
        Test_acc_exit.append(float(acc))
        Test_perc_taken.append(len(good_confidence_indices)/len(test_subset))
        #print("the percentage of the exited data is ", len(good_confidence_indices)/len(test_subset))
        #print("the accuracy of this exit is: ", float(acc))
        #print("we are done checking this level")
        avg3 += (len(good_confidence_indices)/len(test_subset)) * (0.0 if math.isnan(acc) else float(acc)) ###############################

        indices = indices_2
        exit_nodes = exit_nodes_2
        test_flatten_1 = test_flatten_2
        y_test_subset_1 = y_test_subset_2
        i += 1 # move to next proportion
      

      if len(indices) > 0 and i+1 == num_exits: # if we have to go through all the levels up to the last exit because some of the data is still classified wrongly
        #print(f"Some of the input data was wrongly classified, let's move to the next level {i+1} where we use {int(proportions[i]*100)}% of the features")
        # process the rest of the data using the next level
        exit_level = i+1  # now we will exit using the next level
        start_nodes_2 = exit_nodes[indices] # the starting nodes
        num_samples = int(test_flatten.shape[1] * proportions[i])
        #print(f"the number of features used in level {i+1}: ", num_samples)
        test_flatten_2 = test_flatten_1[indices]
        test_subset_2 = test_flatten_1[indices,:num_samples]
        y_test_subset_2 = y_test_subset_1[indices]
        #print(f"the data size for level {i+1}: ", len(test_subset_2))
        predictions_2, exit_nodes_2, prob_2 = clf.predict(test_subset_2, n_classes, exit_level, start_nodes=start_nodes_2)


        # check entropy
        entropy_2 = entropy(prob_2) # a list of entropies
        # check accuracy
        acc = accuracy(y_test_subset_2, predictions_2)
        Test_acc_exit.append(float(acc))
        Test_perc_taken.append(len(predictions_2)/len(test_subset))
        #print("the percentage of the exited data is ", len(predictions_2)/len(test_subset))
        #print("the accuracy of this exit is: ", float(acc))
        #print("we are done checking this level")
        avg3 += (len(predictions_2)/len(test_subset)) * (0.0 if math.isnan(acc) else float(acc)) ###############################

      Total_test_accuracy.append(avg3)
      # now Test_acc_exit and Test_perc_taken and Total_test_accuracy are ready
      #print("exits testing accuracies: ", Test_acc_exit)
      #print("percentages of exited data at each exit level: ", Test_perc_taken)
      #print("total averaged testing accuracy: ", avg3)

      ##########################################################################################################################################

      # calculate the total number of exited data for each exit
      #len_max = max(max(len(T_perc_taken),len( V_perc_taken)),len(Test_perc_taken))
      len_max = num_exits

      check = False
      while (check==False):
        if len(T_perc_taken) < len_max:
          T_perc_taken.append(0)

        if len(V_perc_taken) < len_max:
          V_perc_taken.append(0)

        if len(Test_perc_taken) < len_max:
          Test_perc_taken.append(0)

        if (len(Test_perc_taken) == len(T_perc_taken)) and  (len(Test_perc_taken) == len(V_perc_taken)) and (len(V_perc_taken) == len_max):
          check = True


      for i in range(len_max):
        r = (len(X_train) / total_num_data) * T_perc_taken[i] + (len(X_val) / total_num_data) * V_perc_taken[i] + (len(X_test) / total_num_data) * Test_perc_taken[i]
        entire_data_perc_taken.append(r)
      

      ##################################################################################################################################################
      # sometimes the whole data will exit one of the early exits, so the corresponding E_THi will not be stored

      #if len(E_TH_col)<num_exits:
      # current_length = len(E_TH_col)
        #th_count = current_length-1   
        #t = E_TH1 + k

        #while(len(E_TH_col)<num_exits):
        # E_TH_col.append(round(t+th_count*0.1,1))
        # current_length = len(E_TH_col)
        # th_count += 1

      ####################################################################################################################################################
      
      #while len(E_TH_col)<4:
      # E_TH_col.append(-1)

      #E_TH1_col.append(E_TH_col[0])
      #E_TH2_col.append(E_TH_col[1])
      #E_TH3_col.append(E_TH_col[2])
      #E_TH4_col.append(E_TH_col[3])
      th_col = [round(x,2) for x in th_combination]
      if len(th_col) < 4:
        th_col.append(-1)

      #########################################################################################

      ##############################################################################################################################
      while len(T_acc_exit)<4:
        T_acc_exit.append(-1)

      while len(V_acc_exit)<4:
        V_acc_exit.append(-1)

      while len(Test_acc_exit)<4:
        Test_acc_exit.append(-1)

      ###########################################################################################################################
      while len(T_perc_taken)<4:
        T_perc_taken.append(-1)

      while len(V_perc_taken)<4:
        V_perc_taken.append(-1)

      while len(Test_perc_taken)<4:
        Test_perc_taken.append(-1)


      ################################################################################################################

      while len(entire_data_perc_taken)<4:
        entire_data_perc_taken.append(-1)

      # now entire_data_perc_taken is ready
      #print("percentages of exited data per exit: ", entire_data_perc_taken)

      ###################################################################################################################
      total_nodes = clf.get_total_nodes()

      dictionary = {"dataset": dataset[0], "num_exits": num_exits_col[0], "max_depth" : depth_col[0], "tree splits": tree_splits_col[0], "data percentages": prop[0],
      "T_acc_1": T_acc[0], "T_acc_2": T_acc[1], "T_acc_3": T_acc[2], "T_acc_4": T_acc[3],
              "V_acc_1": V_acc[0], "V_acc_2": V_acc[1], "V_acc_3": V_acc[2], "V_acc_4": V_acc[3],
              "Test_acc_1": Test_acc[0], "Test_acc_2": Test_acc[1], "Test_acc_3": Test_acc[2], "Test_acc_4": Test_acc[3],
              "E_TH1": th_col[0], "E_TH2": th_col[1], "E_TH3": th_col[2], "E_TH4": th_col[3],
              "T_acc_exit_1": T_acc_exit[0], "T_acc_exit_2": T_acc_exit[1], "T_acc_exit_3": T_acc_exit[2], "T_acc_exit_4": T_acc_exit[3],
      "V_acc_exit_1": V_acc_exit[0], "V_acc_exit_2": V_acc_exit[1], "V_acc_exit_3": V_acc_exit[2], "V_acc_exit_4": V_acc_exit[3],
      "Test_acc_exit_1": Test_acc_exit[0], "Test_acc_exit_2": Test_acc_exit[1], "Test_acc_exit_3": Test_acc_exit[2], "Test_acc_exit_4": Test_acc_exit[3],
      "T_perc_taken_1": T_perc_taken[0], "T_perc_taken_2": T_perc_taken[1], "T_perc_taken_3": T_perc_taken[2], "T_perc_taken_4": T_perc_taken[3],
      "V_perc_taken_1": V_perc_taken[0], "V_perc_taken_2": V_perc_taken[1], "V_perc_taken_3": V_perc_taken[2], "V_perc_taken_4": V_perc_taken[3],
      "Test_perc_taken_1": Test_perc_taken[0], "Test_perc_taken_2": Test_perc_taken[1], "Test_perc_taken_3": Test_perc_taken[2], "Test_perc_taken_4": Test_perc_taken[3],
      "entire_data_perc_taken_1": entire_data_perc_taken[0], "entire_data_perc_taken_2": entire_data_perc_taken[1],
      "entire_data_perc_taken_3": entire_data_perc_taken[2], "entire_data_perc_taken_4": entire_data_perc_taken[3],
      "Total_train_accuracy": Total_train_accuracy[0], "Total_validation_accuracy": Total_validation_accuracy[0], "Total_test_accuracy": Total_test_accuracy[0], "total nodes": total_nodes}

      list_of_rows.append(dictionary)


    return list_of_rows

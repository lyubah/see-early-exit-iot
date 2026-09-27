
import numpy as np
from sklearn.utils.multiclass import unique_labels
import math
import itertools
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels 

class RunInference:
    def __init__(self, X_test, y_test, models , stages):
        
        self.X_test = X_test
        self.y_test = y_test
        self.trees_ = None
        self.n_samples = self.X_test.shape[0]
        self.classes_ = None
        self.models = models
        self.stages = stages
        self.srf_entropy = None 
        self._final_passed_indices_after_check_exit = [] 
        self._all_exited_indices_after_check_exit = set() 

        self.indices_in = []
        self.indices_out = []
    
    def entropy(self , probabilities):  
        epsilon = 1e-5  # to avoid taking the logarithm of zero
        return -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)

    def predict_proba(self):

        self.classes_ = unique_labels(self.y_test)
        self.n_classes_ = len(self.classes_)
        n_features_total = self.X_test.shape[1]
        self.all_scores = np.zeros((self.n_samples, self.n_classes_))
        self.all_entropy = []
        self.all_predict = []
        self.connected_prediction = []
        
        for j in range(len(self.models)):
        
            n_features_to_select = int(np.ceil(n_features_total * self.stages[j])) # Use ceil to ensure at least 1
            n_features_to_select = min(n_features_to_select, n_features_total) # Cap
            selected_feature_indices = np.arange(n_features_to_select)
        
            X_test_temp = self.X_test[:, selected_feature_indices]
        
            srf = self.models[j]['model']
            self.trees_ = srf.trees_ # Assuming srf.trees_ is populated by srf.fit()
            
            scores = np.zeros((self.n_samples, self.n_classes_))
            
            for i, tree in enumerate(self.trees_):
                # Assuming tree.predict is available and works on X_test_temp
                predictions = tree.predict(X_test_temp) 
                for sample_idx in range(self.n_samples):
                    predicted_class = predictions[sample_idx]
                    if predicted_class in self.classes_:
                        class_idx = np.where(self.classes_ == predicted_class)[0][0]
                        scores[sample_idx, class_idx] += 1

            self.all_scores += scores
            
            self.connected_prediction.append(np.argmax(self.all_scores , axis=1))
            
            denominator = len(self.trees_) * (j + 1)
            
            proba = self.all_scores / denominator
            
            self.srf_entropy =  self.entropy(proba)
            
            self.all_entropy.append( self.srf_entropy)
            self.all_predict.append(np.argmax(scores , axis=1))
        
        return self.all_entropy , self.all_predict
            
    def max_entropy(self , num_classes, first_class_prob=0.7):
        # math is imported at the top
        probabilities = [first_class_prob] + [(1 - first_class_prob) / (num_classes - 1)] * (num_classes - 1)
        entropy_sum = sum(p * math.log(p) for p in probabilities)
        max_entropy = - entropy_sum
        return max_entropy
    
    def func_threshold_combinations(self , num_exits, dataset_max_entropy):
        # itertools is imported at the top
        # import itertools # Already imported at the top
        num_bins = 4
        bin_width = dataset_max_entropy / num_bins
        thresholds = [(i + 1) * bin_width for i in range(num_bins)]
        
        combinations_set = set() 
        combinations = []
        for combo in itertools.product(thresholds, repeat=num_exits):
            combo_list = list(combo)
            max_threshold = max(combo_list)
            combo_list[0] = max_threshold
            combinations_set.add(tuple(combo_list)) 
        combinations = [list(combo) for combo in combinations_set]
        return combinations[:10]
            
    def generate_all_configurations(self, dataset_name , stages_to_run):
        exits = []
        exits.append((len(stages_to_run)) )
        for num_exits in exits:
            # Determine n_classes based on dataset_name
            n_classes = 0
            if dataset_name == 'Shoaib':
                n_classes = 7
            elif dataset_name == 'Epilepsy':
                n_classes = 4
            elif dataset_name == 'EMGPhysical':
                n_classes = 4
            elif dataset_name == 'SelfRegulationSCP1':
                n_classes = 2
            elif dataset_name == 'WESADchest':
                n_classes = 3
            elif dataset_name == 'PAMAP2':
                n_classes = 5
            else:
                print(f"Warning: Unknown dataset name '{dataset_name}'. Defaulting to 2 classes for entropy calculation.")
                n_classes = 2 # Default to 2 classes if dataset name is not recognized

            entropy_max = self.max_entropy(n_classes)  
            self.th_combinations = self.func_threshold_combinations(num_exits, entropy_max)
                
        return self.th_combinations
    
    
    
    def ExitAtAllStage(self):
        
        accuracies_exit_all = {}
        
        for j , prediction_per_stage in enumerate(self.connected_prediction):
            
            accuracy = accuracy_score(self.y_test, prediction_per_stage)
            accuracies_exit_all[f"accuracy_exit_all_Sample_RF {j+1}"] = f"{accuracy:.4f}"
        
        return accuracies_exit_all

       
    def check_exit(self , sub_rf_entropy , threshold_list_of_keys , predictions , y_test):
        
        if sub_rf_entropy is None:
            print("Error in check_exit: sub_rf_entropy is not available.")
            print("Please ensure predict_proba() has been called and successfully populated sub_rf_entropy.")
            self.indices_in = []
            self.indices_out = []
            return [], [], [] # Return empty list for metrics

        if len(sub_rf_entropy) == 0:
            print("Warning in check_exit: sub_rf_entropy is empty. No samples to check.")
            self.indices_in = []
            self.indices_out = []
            return [], [], [] # Return empty list for metrics

        num_total_samples = len(sub_rf_entropy[0])
        all_original_indices = list(range(num_total_samples))

        print(f"check_exit: Initial number of samples: {num_total_samples}\n")

        passed_indices_for_each_key = [] 
        self._all_exited_indices_after_check_exit = set()
        
        # This will store a list of dictionaries, one for each threshold key
        all_keys_inference_metrics = [] 

        for key_idx, key_specific_thresholds in enumerate(threshold_list_of_keys):
            threshold_strings = [f"{t:.4f}" for t in key_specific_thresholds]
            print(f"--- Processing Key {key_idx + 1} (Thresholds: {threshold_strings}) ---")
            print(f"    (Starting with initial {num_total_samples} samples for this key)")

            indices_being_processed_for_this_key = list(all_original_indices) 
            exited_samples_this_key_cumulative = set()

            # Dictionary to store inference metrics for the CURRENT key
            current_key_inference_metrics = {}
            current_key_inference_metrics["Threshold_Configuration"] = str(key_specific_thresholds) # No Key_X suffix here

            EnergyUsed_sum = []
            Total_acc_per_config = []
            
            for threshold_j_idx, threshold_value in enumerate(key_specific_thresholds):
                
                is_last = (threshold_j_idx ==(len(key_specific_thresholds) - 1))
                
                # Add the threshold value for this specific RF stage within this key
                current_key_inference_metrics[f"Threshold_Value_RF_{threshold_j_idx+1}"] = f"{threshold_value:.4f}" # No Key_X suffix

                if is_last:  # FIX THE CORRECT INDIXES
                    sklearn_accuracy = accuracy_score(y_test[passed_this_stage], predictions[threshold_j_idx][passed_this_stage])  #????????????
                    print(f"Accuracy for the last subset {threshold_j_idx+1}: {sklearn_accuracy:.4f}")
                    
                    exit_percentage = float( len(passed_this_stage)/ num_total_samples)
                    Total_acc_per_config.append(sklearn_accuracy * exit_percentage)  #????/
                    EnergyUsed_sum.append(float(exit_percentage * self.stages[threshold_j_idx]))
                    
                    # Capture metrics for the last stage as well
                    current_key_inference_metrics[f"Accuracy_RF_{threshold_j_idx+1}"] = f"{sklearn_accuracy:.4f}"
                    current_key_inference_metrics[f"Samples_Exited_RF_{threshold_j_idx+1}"] = len(indices_being_processed_for_this_key) # All remaining exit here
                    current_key_inference_metrics[f"Samples_Remaining_RF_{threshold_j_idx+1}"] = 0
                    current_key_inference_metrics[f"Exit_Percentage_RF_{threshold_j_idx+1}"] = f"{exit_percentage:.4f}"
                    
                    # Entropy values for the last stage would be for the remaining samples
                    # entropy_values_for_last_stage = sub_rf_entropy[threshold_j_idx][indices_being_processed_for_this_key]
                    # current_key_inference_metrics[f"Entropy_Values_RF_{threshold_j_idx+1}"] = ';'.join(map(str, entropy_values_for_last_stage)) if len(entropy_values_for_last_stage) > 0 else ''
                    
                else:
                    if not indices_being_processed_for_this_key:
                        exit_percentage = float(len(exited_at_this_stage)/ num_total_samples)

                        print(f"  Key {key_idx+1}, exit stage {threshold_j_idx + 1} (Threshold: {threshold_value:.4f}):")
                        print(f"    No samples left to process from previous stage within this key. Skipping.")
                        # Still add placeholder entries for consistency
                        current_key_inference_metrics[f"Samples_Exited_RF_{threshold_j_idx+1}"] = 0
                        current_key_inference_metrics[f"Samples_Remaining_RF_{threshold_j_idx+1}"] = 0
                        current_key_inference_metrics[f"Accuracy_RF_{threshold_j_idx+1}"] = ''
                        current_key_inference_metrics[f"Exit_Percentage_RF_{threshold_j_idx+1}"] = f"{exit_percentage:.4f}"

                        # current_key_inference_metrics[f"Entropy_Values_RF_{threshold_j_idx+1}"] = ''
                        break 

                    exited_at_this_stage = []
                    passed_this_stage = []
                    
                    for original_sample_index in indices_being_processed_for_this_key:
                        sample_entropy_value = sub_rf_entropy[threshold_j_idx][original_sample_index]
                        
                        if sample_entropy_value < threshold_value:
                            exited_at_this_stage.append(original_sample_index)
                            self._all_exited_indices_after_check_exit.add(original_sample_index) 
                            exited_samples_this_key_cumulative.add(original_sample_index) 
                        else:
                            passed_this_stage.append(original_sample_index)

                    subset_predictions = predictions[threshold_j_idx][exited_at_this_stage]
                    subset_true_labels = y_test[exited_at_this_stage]
                                    
                    temp_indices_processed = indices_being_processed_for_this_key
                    indices_being_processed_for_this_key = passed_this_stage 
                    exit_percentage = float(len(exited_at_this_stage)/ num_total_samples)
                    
                    
                    EnergyUsed_sum.append(float(exit_percentage * self.stages[threshold_j_idx]))
                    
                    print(f"  Key {key_idx+1}, exit stage {threshold_j_idx+1} (Threshold Value: {threshold_value:.4f}):")
                    print(f"    samples were exit : {len(exited_at_this_stage)}")
                    print(f"    samples were passed to next (stage/forest within this key) : {len(indices_being_processed_for_this_key)}")
             
                    
                    if len(subset_predictions) == 0:
                        sklearn_accuracy = -1
                        print(f"\nAccuracy for the subset: No samples in sub rf {threshold_j_idx+1} to calculate accuracy.")
                    else:
                        sklearn_accuracy = accuracy_score(subset_true_labels, subset_predictions)
                        print(f"Accuracy for the subset {threshold_j_idx+1}: {sklearn_accuracy:.4f}")
                        
                    Total_acc_per_config.append(sklearn_accuracy * exit_percentage ) #???
                    
                    # Store metrics for this specific RF stage within this key
                    current_key_inference_metrics[f"Samples_Exited_RF_{threshold_j_idx+1}"] = len(exited_at_this_stage)
                    current_key_inference_metrics[f"Samples_Remaining_RF_{threshold_j_idx+1}"] = len(indices_being_processed_for_this_key)
                    current_key_inference_metrics[f"Accuracy_RF_{threshold_j_idx+1}"] = f"{sklearn_accuracy:.4f}"
                    
                    current_key_inference_metrics[f"Exit_Percentage_RF_{threshold_j_idx+1}"] = f"{exit_percentage:.4f}"
                    # entropy_values_for_stage = sub_rf_entropy[threshold_j_idx][exited_at_this_stage]
                    # current_key_inference_metrics[f"Entropy_Values_RF_{threshold_j_idx+1}"] = ';'.join(map(str, entropy_values_for_stage)) if len(entropy_values_for_stage) > 0 else ''
                    
            current_key_inference_metrics[f"Energy_USED"] = (sum(EnergyUsed_sum))       
            current_key_inference_metrics[f"Total_acc"]  = (sum(Total_acc_per_config))
            # After all thresholds for the current key are processed, add this key's metrics to the list
            all_keys_inference_metrics.append(current_key_inference_metrics)

            passed_indices_for_each_key.append(set(indices_being_processed_for_this_key))
            
            print(f"--- Finished Key {key_idx + 1} ---")
            print(f"  Samples passed ALL thresholds of this Key: {len(indices_being_processed_for_this_key)}")
            print(f"  Total samples exited during this Key's processing: {len(exited_samples_this_key_cumulative)}\n")
            
            if key_idx < len(threshold_list_of_keys) -1 :
                 print("finished one threshold list (key), processing next list (key) with initial samples...")
            else:
                 print("finished all threshold lists (keys).")
                 
        # After all keys are processed, determine final 'in' and 'out'
        if not passed_indices_for_each_key:
            self._final_passed_indices_after_check_exit = []
        else:
            final_passed_set = set(passed_indices_for_each_key[0])
            for i in range(1, len(passed_indices_for_each_key)):
                final_passed_set.intersection_update(passed_indices_for_each_key[i])
            self._final_passed_indices_after_check_exit = sorted(list(final_passed_set))
        
        self.indices_in = self._final_passed_indices_after_check_exit
        self.indices_out = sorted(list(self._all_exited_indices_after_check_exit))

        print("="*30)
        print("Overall Results from check_exit:")
        print(f"Total unique samples that exited at any point (final 'out'): {len(self.indices_out)}")
        print(f"Total samples that passed all keys (final 'in'): {len(self.indices_in)}")
        print("="*30)
        
        return self.indices_in, self.indices_out, all_keys_inference_metrics


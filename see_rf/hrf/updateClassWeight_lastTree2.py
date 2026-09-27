import numpy as np
import pandas as pd
from ReadFile import LoadData
from sklearn.metrics import accuracy_score
import time
from Seq_RandomForest import SequentialRandomForest
from Inference2 import RunInference
import os

def run_staged_srf(X_train, y_train, X_VAL, y_VAL, config_index,
                   overall_max_depth, overall_n_estimators, stages, random_state=42, logger=None):

    current_config_training_metrics = {}
    final_weights = None
    n_features_total = X_train.shape[1]
    last_stage_model = None

    models_info = []

    if logger: logger.info(f"\n--- Starting Staged sRF Learning for Config {config_index} ---")
    start_time_total = time.time()

    for i, percentage in enumerate(stages):
        stage_num = i + 1
        if logger: logger.info(f"\n--- Stage {stage_num} ({int(percentage*100)}% Initial Features) ---")
        start_time_stage = time.time()

        # --- Select Features: Take the first N% ---
        n_features_to_select = int(np.ceil(n_features_total * percentage))
        n_features_to_select = min(n_features_to_select, n_features_total)

        selected_feature_indices = np.arange(n_features_to_select)
        if logger: logger.info(f"  Using first {n_features_to_select} features (Indices 0 to {n_features_to_select-1}).")

        current_X_train_stage = X_train[:, selected_feature_indices]
        current_X_VAL_stage = X_VAL[:, selected_feature_indices]
        if logger: logger.info(f"  Training data shape for stage: {current_X_train_stage.shape}")

        srf_stage = SequentialRandomForest(
            n_estimators=overall_n_estimators,
            random_state=random_state,
            max_depth=overall_max_depth,
            initial_class_weights=final_weights
        )

        # Train sRF for this stage
        if logger: logger.info(f"  Training sRF with {overall_n_estimators} estimators...")
        srf_stage.fit(current_X_train_stage, y_train)
        if logger: logger.info(f" *** number of trees: {len(srf_stage.trees_)}")

        # Evaluate
        y_pred_stage = srf_stage.predict(current_X_VAL_stage)
        acc_stage = accuracy_score(y_VAL, y_pred_stage)
        if logger: logger.info(f"  Stage {stage_num} Test Accuracy: {acc_stage:.4f}")

        # Get final weights for the next stage
        final_weights = srf_stage.get_final_class_weights()
        if logger: logger.info(f"  Final class weights for next stage: { {k: round(v, 3) for k, v in final_weights.items()} }")

        models_info.append({
            'model': srf_stage,
            'features_indices': selected_feature_indices
        })
        end_time_stage = time.time()
        if logger: logger.info(f"  Stage {stage_num} duration: {end_time_stage - start_time_stage:.2f} seconds")

        # Collect training metrics for the current stage into the local dictionary
        current_config_training_metrics[f"tree_splits"] = stages
        current_config_training_metrics[f"num_of_exits"] = len(stages)
        current_config_training_metrics[f"train-acc-{stage_num}"] = f"{acc_stage:.4f}"


    end_time_total = time.time()
    if logger: logger.info(f"\n--- Staged Learning Complete for Config {config_index} ---")
    if logger: logger.info(f"Total duration for Config {config_index}: {end_time_total - start_time_total:.2f} seconds")
    current_config_training_metrics["Model_Size"] = sum(srf_stage.node_counts)

    return models_info, current_config_training_metrics


def Run_orchestrator(dataset_name, max_depth, overall_n_estimators , split_points, config_index, logger):
    """
    Orchestrates a full experiment run (data loading, training, inference)
    for a single configuration and dataset.
    Returns a list of dictionaries, where each dictionary is a row of combined metrics.
    """

    # Load data for the current dataset
    classData = LoadData()
    classData.Read(dataset_name)
    classData.SplitData()

    # Calculate overall dataset length and split percentages once
    total_dataset_length = classData.GetWindow()
    len_train = len(classData.GetTrainX())
    len_val = len(classData.GetValX())
    len_test = len(classData.GetTestX())

    # Calculate weights based on dataset split sizes
    if total_dataset_length == 0:
        weight_train = 0.0
        weight_val = 0.0
        weight_test = 0.0
        if logger: logger.warning("Total dataset length is zero. Weights for overall accuracy/energy will be zero.")
    else:
        weight_train = len_train / total_dataset_length
        weight_val = len_val / total_dataset_length
        weight_test = len_test / total_dataset_length

    split_weights = {
        "Train": weight_train,
        "Val": weight_val,
        "Test": weight_test
    }
    if logger: logger.info(f"Calculated split weights: {split_weights}")

    # Run staged sRF training
    all_model, current_config_training_metrics = run_staged_srf(
        classData.GetTrainX(), classData.GetYtrain(), classData.GetValX(), classData.GetYval(),
        config_index=config_index,
        overall_max_depth=max_depth,
        overall_n_estimators= overall_n_estimators,
        stages=split_points,
        random_state=42,
        logger=logger
    )

    results_to_return = []
    datasets_for_inference = {
        "Train": (classData.GetTrainX(), classData.GetYtrain()),
        "Val": (classData.GetValX(), classData.GetYval()),
        "Test": (classData.GetTestX(), classData.GetYtest())
    }

    # Generate threshold combinations once, as they are part of the configuration
    inference_obj_for_thresholds = RunInference(X_test=classData.GetTestX(), y_test=classData.GetYtest(), models=all_model, stages=split_points)
    th_combinations = inference_obj_for_thresholds.generate_all_configurations(dataset_name, split_points)

    # Loop through each threshold key's inference metrics and prepare a row for each
    for key_idx, key_specific_thresholds in enumerate(th_combinations):
        # Create a new base row dictionary for each threshold configuration (key)
        combined_config_row = {
            "Config_Index": config_index,
            "Overall_Max_Depth": max_depth,
            "Num_Trees_per_RF": overall_n_estimators,
            "Overall_Trees" : overall_n_estimators * len(split_points) ,
            **current_config_training_metrics,
            "Threshold_Configuration": str(key_specific_thresholds),
            **{f"Threshold_Value_RF_{j+1}": f"{t:.4f}" for j, t in enumerate(key_specific_thresholds)}
        }

        # Now, loop through each data type for inference and add type-specific metrics
        for data_type, (X_data, y_data) in datasets_for_inference.items():
            if logger: logger.info(f"\n--- Running Inference for {data_type} Data with Thresholds: {key_specific_thresholds} ---")
            inference_obj = RunInference(X_test=X_data, y_test=y_data, models=all_model, stages=split_points)

            sub_forest_entropy, prediction = inference_obj.predict_proba()

            _, _, all_keys_inference_metrics_list_from_check_exit = inference_obj.check_exit(
                sub_forest_entropy, [key_specific_thresholds], prediction, y_data
            )
            key_inference_data_for_this_data_type = all_keys_inference_metrics_list_from_check_exit[0] if all_keys_inference_metrics_list_from_check_exit else {}

            all_stage_exit_accuracies = inference_obj.ExitAtAllStage()

            # Add *only* the data-type specific inference metrics, with data_type suffix
            for key, value in key_inference_data_for_this_data_type.items():
                if not (key == "Threshold_Configuration" or key.startswith("Threshold_Value_RF_")):
                    combined_config_row[f"{key}_{data_type}"] = value

            for key, value in all_stage_exit_accuracies.items():
                combined_config_row[f"{key}_{data_type}"] = value

        # --- Calculate Weighted Overall Exit Percentage for each RF stage across all data types ---
        for stage_num in range(1, len(split_points) + 1):
            weighted_overall_exit_percentage_for_stage = 0.0
            total_weight_applied_percentage = 0.0
            
            for data_type, weight in split_weights.items():
                percentage_key = f"Exit_Percentage_RF_{stage_num}_{data_type}"
                
                exit_pct_val = combined_config_row.get(percentage_key)
                
                if exit_pct_val is not None and exit_pct_val != '':
                    try:
                        exit_pct_float = float(exit_pct_val)
                        weighted_overall_exit_percentage_for_stage += (exit_pct_float * weight)
                        total_weight_applied_percentage += weight
                    except ValueError:
                        if logger: logger.warning(f"Could not convert '{exit_pct_val}' for {percentage_key} to float. Skipping for weighted overall percentage.")
                        pass
            
            if total_weight_applied_percentage > 1e-9:
                combined_config_row[f"Weighted_Overall_Exit_Percentage_RF_{stage_num}"] = f"{weighted_overall_exit_percentage_for_stage:.4f}"
            else:
                combined_config_row[f"Weighted_Overall_Exit_Percentage_RF_{stage_num}"] = ''

        # --- Calculate Total Weighted Energy Used ---
        total_weighted_energy = 0.0
        total_weight_applied_energy = 0.0

        total_weighted_acc = 0.0
        total_weight_applied_acc = 0.0

        for data_type, weight in split_weights.items():
            energy_key = f"Energy_USED_{data_type}"
            accuracy_key =f"Total_acc_{data_type}"
            energy_val = combined_config_row.get(energy_key)
            accuracy_val = combined_config_row.get(accuracy_key)
            
            if energy_val is not None and energy_val != '':
                try:
                    energy_float = float(energy_val)
                    total_weighted_energy += (energy_float * weight)
                    total_weight_applied_energy += weight
                except ValueError:
                    if logger: logger.warning(f"Could not convert '{energy_val}' for {energy_key} to float. Skipping for total weighted energy.")
                    pass
        
        
            if accuracy_val is not None and accuracy_val != '':
                try:
                    accuarcy_float = float(accuracy_val)
                    total_weighted_acc += (accuarcy_float * weight)
                    total_weight_applied_acc += weight
                except ValueError:
                    if logger: logger.warning(f"Could not convert '{accuracy_val}' for {accuracy_key} to float. Skipping for total weighted energy.")
                    pass
        
        
            
        if total_weight_applied_energy > 1e-9:
            combined_config_row["data_total_Energy"] = f"{total_weighted_energy:.4f}"

        if total_weight_applied_acc > 1e-9:
            combined_config_row["total_accurcy"] = f"{total_weighted_acc:.4f}"
    



        results_to_return.append(combined_config_row)

    if logger: logger.info(f"============================================================")
    if logger: logger.info(f"Finished model for Configuration #{config_index}\n\n")
    if logger: logger.info(f"============================================================ \n\n")

    return results_to_return

# This part is for standalone testing if needed, not used by run_parallel.py
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
    
    # Example standalone run
    datasetName = 'Epilepsy'
    split_points_example = [0.2, 0.4, 1.0]
    max_depth_example = 500
    config_index_example = 1
    
    print(f"Running orchestrator for {datasetName} (Config {config_index_example}) in standalone mode...")
    results = Run_orchestrator(dataset_name=datasetName, max_depth=max_depth_example,
                               overall_n_estimators=100, split_points=split_points_example,
                               config_index=config_index_example, logger=logging)
    
    if results:
        print(f"Orchestrator returned {len(results)} rows.")
    else:
        print("Orchestrator returned no results.")
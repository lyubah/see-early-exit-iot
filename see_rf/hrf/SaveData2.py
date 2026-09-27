import csv
import os
import numpy as np

class SaveToCSV:
    def __init__(self, DataSet_Name , combined_filename="_records.csv"):
        self.dataset_name = DataSet_Name
        os.makedirs("RecordsFolder", exist_ok=True)
        self.combined_filename = os.path.join("RecordsFolder", f"{self.dataset_name}{combined_filename}")
        self.all_combined_headers_union = set() # To keep track of all unique headers across all configurations
        self.combined_rows_buffer = [] # Stores all complete rows (dictionaries) for all configurations

    def append_completed_config_row(self, config_data_dict):
        """
        Appends a completed row (training + inference for one key) to the buffer.
        This method is called once per (configuration, threshold key) pair.
        """
        config_data_dict["DataSet_Name"] = self.dataset_name
        # Update the union of all headers seen so far
        self.all_combined_headers_union.update(config_data_dict.keys())
        
        self.combined_rows_buffer.append(config_data_dict)
        # print(f"Buffered combined metrics for a (config, key) pair.")

    def write_all_combined_metrics_to_file(self):
        """
        Writes all collected configuration rows (training + inference) to the CSV file.
        This should be called once at the very end of all configurations.
        """
        # Define a desired order for base headers
        base_header_order = [
            'Config_Index',
            'DataSet_Name',
            'Overall_Max_Depth',
            'Num_Trees_per_RF', # Renamed from 'Overall_Num_Trees_per_RF'
            'Overall_Trees', # NEW: Overall_Trees
            'Model_Size',
            'tree_splits', # Training specific
            'num_of_exits', # Training specific
            'Threshold_Configuration' # Configuration specific
        ]

        # Filter and sort training accuracies
        train_acc_headers = sorted([h for h in self.all_combined_headers_union if h.startswith("train-acc-")],
                                   key=lambda x: int(x.split('-')[-1]))
        
        # Filter and sort Threshold_Value_RF headers (these should NOT have _Train/_Val/_Test suffixes)
        threshold_value_headers = sorted([
            h for h in self.all_combined_headers_union
            if h.startswith("Threshold_Value_RF_") and not any(h.endswith(suffix) for suffix in ['_Train', '_Val', '_Test'])
        ], key=lambda x: int(x.split('_')[-1]))

        # Collect all other dynamic headers (which should mostly be the suffixed inference metrics and overall accuracies)
        dynamic_headers = sorted([
            h for h in self.all_combined_headers_union
            if h not in base_header_order and not h.startswith("train-acc-")
            and not h.startswith("Threshold_Value_RF_") # already handled
        ], key=lambda x: self._get_combined_header_sort_key(x))

        final_headers = base_header_order + train_acc_headers + threshold_value_headers + dynamic_headers

        
        print(f"Writing all combined metrics for dataset {self.dataset_name} to {self.combined_filename}...")
        with open(self.combined_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(final_headers) # Write the comprehensive header

            for row_data in self.combined_rows_buffer:
                # Ensure all rows have values for all columns in final_headers
                # Fill missing columns with empty strings
                row_to_write = [row_data.get(header, '') for header in final_headers]
                writer.writerow(row_to_write)
        print(f"All combined metrics (across all configurations) saved to {self.combined_filename}")

    def _get_combined_header_sort_key(self, header):
        # Custom sort key for combined headers
        # Sorts by main category (training vs inference), then by metric type, then by stage/RF number, then by data split type
        data_split_order = {'Train': 0, 'Val': 1, 'Test': 2, '': 99} # Order for suffixes

        # Extract data split type if present
        data_split_type = ''
        header_base = header
        for ds_type_suffix in ['_Train', '_Val', '_Test']:
            if header.endswith(ds_type_suffix):
                data_split_type = ds_type_suffix[1:] # Remove leading underscore
                header_base = header[:-len(ds_type_suffix)]
                break

        ds_order = data_split_order.get(data_split_type, 99)

        try:
            # --- Training Metrics Groups (Group 0) ---
            if 'Config_Index' == header_base:
                return 0, 0, 0, ds_order
            if 'DataSet_Name' == header_base:
                return 0, 1, 0, ds_order
            if 'Overall_Max_Depth' == header_base:
                return 0, 2, 0, ds_order
            if 'Num_Trees_per_RF' == header_base: # Existing
                return 0, 3, 0, ds_order
            if 'Overall_Trees' == header_base: # NEW: Sort this after Overall_Num_Trees_per_RF
                return 0, 3, 1, ds_order
            if 'Model_Size' == header_base:
                return 0, 4, 0, ds_order
            if 'tree_splits' == header_base:
                return 0, 5, 0, ds_order
            if 'num_of_exits' == header_base:
                return 0, 5, 1, ds_order
            if 'train-acc-' in header_base:
                stage_num = int(header_base.split('-')[-1])
                return 0, 6, stage_num, ds_order

            # --- Inference Configuration Headers (Group 1, without data_split_type suffix) ---
            if 'Threshold_Configuration' == header_base:
                return 1, 0, 0, 0
            if 'Threshold_Value_RF_' in header_base and not data_split_type:
                rf_num = int(header_base.split('_')[3])
                return 1, 1, rf_num, 0

            # --- Inference Metrics Groups (Group 2, with data_split_type suffix) ---
            if data_split_type:
                if 'Samples_Exited_RF_' in header_base:
                    rf_num = int(header_base.split('_')[3])
                    return 2, 0, rf_num, ds_order
                if 'Samples_Remaining_RF_' in header_base:
                    rf_num = int(header_base.split('_')[3])
                    return 2, 1, rf_num, ds_order
                if 'Accuracy_RF_' in header_base:
                    rf_num = int(header_base.split('_')[2])
                    return 2, 2, rf_num, ds_order

                if 'Exit_Accuracy_RF_' in header_base:
                    rf_num = int(header_base.split('_')[2])
                    return 2, 3, rf_num, ds_order

                if 'accuracy_exit_all_' in header_base:
                    stage_num = int(header_base.split('_')[-1])
                    return 2, 4, stage_num, ds_order

                if 'Energy_USED' == header_base:
                    return 2, 5, 0, ds_order
                
                if 'Total_acc' == header_base:
                    return 2, 6, 0, ds_order

            # --- Overall Exit Metrics (Group 3, no data_split_type suffix) ---
            if 'Weighted_Overall_Exit_Percentage_RF_' in header_base and not data_split_type:
                rf_num = int(header_base.split('_')[-1])
                return 3, 0, rf_num, 0
            
            if 'data_total_Energy' == header_base:
                return 3, 1, 0, 0

        except (ValueError, IndexError):
            pass
        return 999, header
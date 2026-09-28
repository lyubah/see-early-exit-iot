import csv
import pickle
from baseline_inference import read_configuration_return_results
import argparse
import os
import time
from sensor_control import initialize_bmi160, auto_calibrate, sensor_on, sensor_sleep




def parse_args():
    parser = argparse.ArgumentParser(description = "Full-window RF baseline on the board (sensor on for every window)")
    parser.add_argument("--dataset_name", type=str, default= "Epilepsy", help = "The Dataset name")
    return parser.parse_args()


def add_header(file, header):
    writer = csv.writer(file)
    writer.writerow(header)


def write_content_to_file(file, content, header): # the content is a list of dictionaries
    writer = csv.writer(file)
    for line in content:
        row = [line[key] for key in header]
        writer.writerow(row)


if __name__ == "__main__":
    
    t1 = time.time()
    time.sleep(5)
    t2 = time.time()
    T = t2-t1 
    print("t1: ", t1)
    print("t2: ", t2)

    args = parse_args()
    

    file_path = f'Datasets/{args.dataset_name}_dataLabels.pkl'
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)
    data = data_dict['data']

    n_window, n_channel, n_data = data.shape

    # window length in SAMPLES per channel at the sensor rate fs_base (Hz);
    # the acquisition time per window is window_len / fs_base seconds
    # (e.g. Epilepsy 206 samples / 250 Hz = 0.82 s)
    window_len = n_data

    if args.dataset_name == 'Shoaib':
        fs_base = 50
        window_len = 200    # 4 s
    if args.dataset_name == 'Epilepsy':
        fs_base = 250
        window_len = 206    # 0.82 s
    if args.dataset_name == 'EMGPhysical':
        fs_base = 100
        window_len = 200    # 2 s
    if args.dataset_name == 'SelfRegulationSCP1':
        fs_base = 256
        window_len = 896    # 3.5 s
    if args.dataset_name == 'WESADchest':
        fs_base = 700
        window_len = 200    # 0.29 s
    if args.dataset_name == 'PAMAP2':
        fs_base = 100
        window_len = 512    # 5.12 s

    initialize_bmi160()
    print("BMI160 Initialized")
    auto_calibrate()


    results = read_configuration_return_results(window_len, fs_base, args.dataset_name,sensor_on)
    os.system("pkill -f data_logger.py")

    os.makedirs('saved_pkl', exist_ok=True)
    output_file = f'saved_pkl/{args.dataset_name}_accuracy_results.csv'
    header = ['t_start','t_end', 'total', 'true_label', 'prediction', 'correctness']
    with open(output_file, "w", newline="") as f1:
        add_header(f1, header)
        write_content_to_file(f1, results, header)
                
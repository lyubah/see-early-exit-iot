import csv
import pickle
from Training_Inference import read_configuration_return_results
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description = "RF-V Inference")
    parser.add_argument("--dataset_name", type=str, default= "Epilepsy", help = "The Dataset name")
    parser.add_argument("--num_exits", type=int, help = "The number of exits")
    parser.add_argument("--proportions", type=float, help = "Data proportions", nargs="+")
    parser.add_argument("--th_combination", type=float, help = "Threshold combination", nargs="+")
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

    args = parse_args()
    #results = read_configuration_return_results(args.dataset_name, args.num_exits, args.proportions, args.th_combination)
    results = read_configuration_return_results(args.dataset_name, args.num_exits, args.proportions, args.th_combination)
    os.system("pkill -f 'python3 data_logger.py'")
    output_file = f'{args.dataset_name}_accuracy_results.csv'
    header = ['t_start','t1', 't2', 't3', 't4', 'total', 'true_label', 'prediction', 'correctness', 'exit_taken', 'data%']
    with open(output_file, "w", newline="") as f1:
        add_header(f1, header)
        write_content_to_file(f1, results, header)
                
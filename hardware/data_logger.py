from ina219 import INA219
import sys
import time
import csv
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str)
args = parser.parse_args()

SHUNT_OHMS = 0.1
ina = INA219(SHUNT_OHMS, busnum=1)
ina.configure()
 
header = ["time s", "power mW"]
filename = f"{args.dataset_name}_shared_Power.csv"
data_all = []
 
 
with open(filename, 'w', newline='') as csvfile:
 
    writer = csv.writer(csvfile)
    writer.writerow(header)
    while True:
        p = ina.power()
        t = time.time()
        print(p)
        #print(ina.voltage())
        data = []
        data.append(t)
        data.append(p)
 
 
        # Write the data rows
        writer.writerow(data)
#        time.sleep(0.05)
 

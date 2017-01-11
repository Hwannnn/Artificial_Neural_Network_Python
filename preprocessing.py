import csv
import pandas as pd
import numpy as np

filename = "wdbc.csv"

dataset = []
for row in csv.reader(open(filename)):
    dataset.append(row)

del dataset[0]

for i in range(len(dataset)):
    dataset[i][2:32] = [float(x) for x in dataset[i][2:32]]
    str(dataset[i][1])

# Dummy Coding
for i in range(len(dataset)):
    del dataset[i][0]
    if dataset[i][0] == "B":
        del dataset[i][0]
        dataset[i].append(1)
        dataset[i].append(0)
    elif dataset[i][0] == "M":
        del dataset[i][0]
        dataset[i].append(0)
        dataset[i].append(1)

dataset = pd.DataFrame(dataset)
temp = np.zeros((569, 32), dtype='float64', order='C')

'''
for c in range(0,30) :
    mean_val = np.mean(dataset.values[:,c])
    std_val = np.std(dataset.values[:,c])
    for r in range(0,dataset.shape[0]) :
        temp[r,c] = (dataset.values[r,c]-mean_val)/(std_val)
dataset = pd.DataFrame(dataset.values)
'''

# MinMax Scale
for c in range(0,32) :
    max_val = max(dataset.values[:,c])
    min_val = min(dataset.values[:,c])
    for r in range(0,dataset.shape[0]) :
        temp[r,c] = (dataset.values[r,c]-min_val)/(max_val-min_val)

temp = pd.DataFrame(temp)

temp.to_csv("wdbc_1_of_c_coding.csv")
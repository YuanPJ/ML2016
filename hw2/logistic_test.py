import sys
import csv
import math
import numpy as np

with open(sys.argv[1], 'r') as csvfile :
    reader = csv.reader(csvfile)
    data = list(reader)
    for line in reader :
        data = list(reader)
    csvfile.close()
par = np.array(data[0], dtype = float)
bias = np.array(data[1], dtype = float)

with open(sys.argv[2], 'r') as csvfile :
    reader = csv.reader(csvfile)
    data = list(reader)
    for line in reader :
        data = list(reader)
    csvfile.close()
mat = np.array(data, dtype = float)
mat = np.delete(mat, 0, axis = 1)
z = np.dot(mat, par) + bias
sig = 1 / (1 + np.exp(-z))
myans = []
bol = sig > 0.5
for i in range(600) :
    if (bol[i]) :
        myans.append(1)
    else :
        myans.append(0)

with open(sys.argv[3], 'w') as csvfile :
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for id in range(1, 601) :
        writer.writerow({'id' : id, 'label' : myans[id-1]})

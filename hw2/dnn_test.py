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

node = np.array(data[0], dtype = int)
w = []
pos = 1
depth = len(node)-1
xx = []

for i in range(1, depth+1) :
    w.append(np.array(data[pos : pos+node[i]], dtype = float))
    pos = pos + node[i]
bias = np.array(data[pos], dtype = float)

with open(sys.argv[2], 'r') as csvfile :
    reader = csv.reader(csvfile)
    data = list(reader)
    for line in reader :
        data = list(reader)
    csvfile.close()
mat = np.array(data, dtype = float)
mat = np.delete(mat, 0, axis = 1)
xx.append(np.transpose(mat))

z = np.dot(w[0], xx[0]) + bias[0]
xx.append(z)
for d in range(1, depth) :
    sig = 1 / (1 + np.exp(-xx[d]))
    z = np.dot(w[d], sig) + bias[d]
    xx.append(z)

myans = []
for i in range(600) :
    if (xx[depth][0, i] > 0.5) :
        myans.append(1)
    else :
        myans.append(0)

with open(sys.argv[3], 'w') as csvfile :
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for id in range(1, 601) :
        writer.writerow({'id' : id, 'label' : myans[id-1]})

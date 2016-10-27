import sys
import csv
import math
import numpy as np
import random as rand

with open(sys.argv[1], 'r') as csvfile :
    reader = csv.reader(csvfile)
    data = list(reader)
    for line in reader :
        data = list(reader)
    csvfile.close()

mat = np.array(data, dtype = float)
ans = mat[:, -1]
mat = mat[:, 1:-1]

# 57 par + 1 bias
ran = []
par = np.random.uniform(-0.1, 0.1, 57)
bias = np.random.rand(1)

lr = 0.075
iter = 10001
G = np.zeros(57)
B = 0
for times in range(iter) :
    z = np.dot(mat, par) + bias
    sig = 1 / (1 + np.exp(-z))
    cross = -(   ans  * np.log(np.maximum(sig, 1e-15)) +
              (1-ans) * np.log(np.maximum(1-sig, 1e-15)) )

    g = -1 * np.dot( (ans - sig), mat)
    G = G + np.square(g)
    b = -1 * 2 * np.sum(ans - sig)
    B = B + b * b

    par = par - lr * g / np.sqrt(G)
    bias = bias - lr * b / np.sqrt(B)

    #loss = np.sum(cross)
    #print(loss)

with open(sys.argv[2], 'w') as csvfile :
    writer = csv.writer(csvfile)
    writer.writerow(par.tolist())
    writer.writerow(bias)

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

node = [57,40,1]
lr = 0.01
iter = 10001

depth = len(node)-1
w = []
G = []
x = []
xx = []
delta = []
bias = np.random.rand(depth)
B = np.zeros(depth, dtype = float)
for i in range(1, len(node)) :
    par = np.random.uniform(-0.01, 0.01, (node[i], node[i-1]))
    w.append(par)
    zero = np.zeros((node[i], node[i-1]), dtype = float)
    G.append(zero)
for i in range(1, len(node)+1) :
    zero = np.zeros((node[i-1], 4001), dtype = float)
    x.append(zero)
    xx.append(zero)
    delta.append(zero)
x[0] = np.transpose(mat)
#papjjjjjjiloveur = np.random.rand(depth, width, 58)

np.set_printoptions(threshold=np.nan)
for times in range(iter) :
    z = np.dot(w[0], x[0]) + bias[0]
    x[1] = z
    for d in range(1, depth) :
        sig = 1 / (1 + np.exp(-x[d]))
        z = np.dot(w[d], sig) + bias[d]
        x[d+1] = z

    sig = 1 / (1 + np.exp(-x[depth]))
    dsig = sig * (1-sig)
    cross = -(    ans * np.log(np.maximum(sig, 1e-15)) +
              (1-ans) * np.log(np.maximum(1-sig, 1e-15)) )
    loss = np.sum(cross)
    delta[depth] = sig - ans

    for d in range(1, depth+1) :
        sig = 1 / (1 + np.exp(-x[depth-d]))
        dsig = sig * (1-sig)
        sum = np.dot(np.transpose(w[depth-d]), delta[depth-d+1])
        delta[depth-d] = dsig * sum

    for d in range(depth) :
        g = np.dot(delta[d+1], np.transpose(x[d]))
        G[d] = G[d] + np.square(g)
        w[d] = w[d] - lr * g / np.sqrt(G[d])
        b = np.sum(delta[d+1])
        B[d] = B[d] + b * b
        bias[d] = bias[d] - lr * b / np.sqrt(B[d])
    #print(loss)

with open(sys.argv[2], 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(node)
    for i in range (len(w)) :
        writer.writerows(w[i].tolist())
    writer.writerow(bias.tolist())


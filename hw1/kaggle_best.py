import csv
import math
import numpy as np
import random as rand

# Read data and construct a 2D array.
with open('data/train.csv', 'r', encoding = 'BIG5') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader :
        data = list(reader)
    csvfile.close()

# Seperate the data into numpy matrix[12][480]
month = 0
day = 0

temp = [ [ [], [], [], [], [], [], [], [], [], [], [], [] ],
         [ [], [], [], [], [], [], [], [], [], [], [], [] ],
         [ [], [], [], [], [], [], [], [], [], [], [], [] ],
         [ [], [], [], [], [], [], [], [], [], [], [], [] ],
         [ [], [], [], [], [], [], [], [], [], [], [], [] ] ]
num = len(temp)

for i in range(len(data)) :
    if ( data[i][2] == 'PM2.5' ) :
        day = day + 1
        temp[0][month].extend( data[i][3:27] ) # PM2.5
        temp[1][month].extend( data[i-2][3:27] ) #O3
        temp[2][month].extend( data[i-3][3:27] ) #NOx
        temp[3][month].extend( data[i-4][3:27] ) #NO2
        temp[4][month].extend( data[i-5][3:27] ) #NO
        if ( day == 20 ) :
            month = month + 1
            day = 0
mat = np.array(temp, dtype=float)

# Generate parameters randomly.
ran = []
for index in range(num) :
    ran.append([])
for i in range(9) :
    for index in range(num) :
        ran[index].append(rand.uniform(0, 1))
par = np.array(ran)
bias = rand.uniform(0, 0.1)

par[0] = [-2.29229551e-02, -3.56201542e-02, 2.24035092e-01,
          -2.32844907e-01, -5.78932084e-02, 5.27511607e-01,
          -5.64240563e-01, -2.97617929e-04, 1.09713633e+00]

# Training data.
lr = 0.000001
iter = 8001
lamda = 0.1
g = np.array([])
b = 0

for times in range(iter) :
    data = mat[:, :, 0:8]
    loss = 0
    for i in range(9, 479) : # first train 0-8, last train 470-478
        ans = mat[0, :, i+1]
        data = np.insert(data, 8, mat[:, :, i], axis = 2)
        myans = [0]
        for index in range(num) :
            out = np.dot(data[index], np.transpose(par[index]))
            myans = myans + out
        myans = myans + bias
        dif = ans - myans
        g = np.dot(dif, data[:, :, 0:9]) * -1
        b = np.sum(2 * dif * -1)
        par[1:num] = par[1:num] - lr * 2 * g[1:num]
        bias = bias - lr * b
        data = np.delete(data, 0, axis = 2)
        loss = loss + np.sum(dif * dif)
    if (times%100 == 0) :
        print('times', times)
        print('par', '= np.', repr(par))
        print('bias', '=', bias)
        print('loss', loss)

# Read in testing data
with open('data/test_X.csv', 'r', encoding = 'BIG5') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader :
        data = list(reader)
    csvfile.close()

pm2 = []
o3 = []
nox = []
no2 = []
no = []
for i in range(len(data)) :
    if ( data[i][1] == 'PM2.5' ) :
        pm2.append(data[i][2:11])
        o3.append(data[i-2][2:11])
        nox.append(data[i-3][2:11])
        no2.append(data[i-4][2:11])
        no.append(data[i-5][2:11])
data = [pm2, o3, nox, no2, no]
num = len(data)
id = np.array(data, dtype = float)

myans = [0]
for index in range(num) :
    out = np.dot(id[index], np.transpose(par[index]))
    myans = myans + out
myans = myans + bias

with open('kaggle_best.csv', 'w') as csvfile:
    fieldnames = ['id', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for id in range(240) :
        writer.writerow({'id': 'id_'+ str(id), 'value': myans[id]})

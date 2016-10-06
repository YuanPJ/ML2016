import csv
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
temp = [[], [], [], [], [], [], [], [], [], [], [], []]
for i in range(len(data)) :
    if ( data[i][2] == 'PM2.5' ) :
        day = day + 1
        temp[month].extend( data[i][3:27] )
        if ( day == 20 ) :
            month = month + 1
            day = 0
mat = np.array(temp, dtype=float)

# Generate 10 parameters randomly.
ran = []
for i in range(10) :
    ran.append(rand.uniform(-0.5, 1))
par = np.array(ran)

# Training data.
lr = 0.0000000001
iter = 10000000

for times in range(iter) :
    pm2 = mat[:, 0:8]
    pm2 = np.insert(pm2, 8, 1, axis = 1)
    for i in range(9, 479) : # first train 0-8, last train 470-478
        pm2 = np.insert(pm2, 8, mat[:, i], axis = 1)
        out = np.dot(pm2, par)
        dif = mat[:, i+1] - out
        par[0:9] = par[0:9] - lr * np.sum(2 * np.dot(dif, pm2[:, 0:9]))
        par[9] = par[9] - lr * np.sum(2 * dif)
        sum = np.sum(dif)
        print('dif', dif)
        print('pm2', pm2)
        print('y', mat[:, i+1])
        print('out', out)
        print('par', par)
        print('sum', np.sum(2 * dif * dif))
        pm2 = np.delete(pm2, 0, axis = 1)
        input()
#print(pm2)
#print(par)

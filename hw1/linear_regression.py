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
         [ [], [], [], [], [], [], [], [], [], [], [], [] ],
         [ [], [], [], [], [], [], [], [], [], [], [], [] ],
         [ [], [], [], [], [], [], [], [], [], [], [], [] ],
         [ [], [], [], [], [], [], [], [], [], [], [], [] ],
         [ [], [], [], [], [], [], [], [], [], [], [], [] ],
         [ [], [], [], [], [], [], [], [], [], [], [], [] ] ]
num = len(temp)

for i in range(len(data)) :
    if ( data[i][2] == 'PM2.5' ) :
        day = day + 1
        temp[0][month].extend( data[i][3:27] )   # PM2.5^2
        temp[1][month].extend( data[i][3:27] )   # PM2.5
        temp[2][month].extend( data[i-1][3:27] ) # PM10
        temp[3][month].extend( data[i-2][3:27] ) # O3
        temp[4][month].extend( data[i-3][3:27] ) # NOx
        temp[5][month].extend( data[i-4][3:27] ) # NO2
        temp[6][month].extend( data[i-5][3:27] ) # NO
        temp[7][month].extend( data[i-6][3:27] ) # NMHC
        temp[8][month].extend( data[i-7][3:27] ) # CO
        temp[9][month].extend( data[i+1][3:27] ) # RN
        if ( day == 20 ) :
            month = month + 1
            day = 0

rain = 9
for month in range(12):
    for i in range(len(temp[0][0])) :
        if (temp[rain][month][i] == 'NR') :
            temp[rain][month][i] = '0.0'

mat = np.array(temp, dtype=float)
mat[0] = np.square(mat[0])
for month in range(12):
    for i in range(len(temp[0][0])) :
        if (mat[7, month, i] == 0) :
            mat[7, month, i] = 0
        else :
            mat[7, month, i] = 1 / mat[7, month, i]

# Generate parameters randomly.
ran = []
for index in range(num) :
    ran.append([])
for i in range(9) :
    for index in range(num) :
        ran[index].append(rand.uniform(0, 1))
par = np.array(ran)
bias = rand.uniform(0, 0.1)

par[0] = [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          -1.18823281e-03,   3.79480330e-04,   4.64721740e-04]
par[1] = [ -2.91894160e-02,  -3.70030734e-02,   2.24342896e-01,
           -2.33749998e-01,  -5.54508372e-02,   5.19824579e-01,
           -4.92923064e-01,  -2.77964939e-02,   1.06607673e+00]

# Set some parameters to zero
par[0, 0] = par[0, 1] = par[0, 2] = par[0, 3] = par[0, 4] = par[0, 5] = 0
par[2, 0] = par[2, 1] = par[2, 2] = par[2, 3] = 0
par[3, 0] = par[3, 1] = par[3, 2] = par[3, 3] = 0
par[rain, 0] = par[rain, 1] = par[rain, 2] = par[rain, 3] = 0
par[rain, 4] = par[rain, 5] = par[rain, 6]= 0
for i in range (3, rain) :
    par[i, 0] = par[i, 1] = par[i, 2] = par[i, 3] = par[i, 4] = 0

# Training data.
lr = 0.05
iter = 8001
lamda = 0.001
g = np.array([])
G = np.zeros((num, 9))
b = 0
B = 0

for times in range(iter) :
    data = mat[:, :, 0:8]
    loss = 0
    for i in range(9, 479) : # first train 0-8, last train 470-478
        ans = mat[1, :, i+1]
        data = np.insert(data, 8, mat[:, :, i], axis = 2)
        myans = [0]
        for index in range(num) :
            out = np.dot(data[index], np.transpose(par[index]))
            myans = myans + out
        myans = myans + bias
        dif = ans - myans
        g = np.dot(dif, data[:, :, 0:9]) * -1 + lamda * 2 * par
        G = G + np.square(g)
        b = np.sum(2 * dif * -1) + lamda * 2 * bias
        B = B + b * b
        temp = par[1]
        par[2:num] = par[2:num] - lr * 2 * g[2:num] / np.sqrt(G)[2:num]
        bias = bias - lr * b / math.sqrt(B)
        # Set some parameters to zero.
        par[0, 0] = par[0, 1] = par[0, 2] = par[0, 3] = par[0, 4] = par[0, 5] = 0
        par[2, 0] = par[2, 1] = par[2, 2] = par[2, 3] = 0
        par[3, 0] = par[3, 1] = par[3, 2] = par[3, 3] = 0
        par[rain, 0] = par[rain, 1] = par[rain, 2] = par[rain, 3] = 0
        par[rain, 4] = par[rain, 5] = par[rain, 6]= 0
        for i in range (4, rain) :
            par[i, 0] = par[i, 1] = par[i, 2] = par[i, 3] = par[i, 4] = 0
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
pm10 = []
o3 = []
nox = []
no2 = []
no = []
nmhc = []
co = []
rn = []
for i in range(len(data)) :
    if ( data[i][1] == 'PM2.5' ) :
        pm2.append(data[i][2:11])
        pm10.append(data[i-1][2:11])
        o3.append(data[i-2][2:11])
        nox.append(data[i-3][2:11])
        no2.append(data[i-4][2:11])
        no.append(data[i-5][2:11])
        nmhc.append(data[i-6][2:11])
        co.append(data[i-7][2:11])
        rn.append(data[i+1][2:11])
for i in range(len(rn)) :
    for j in range(len(rn[0])) :
        if (rn[i][j] == 'NR') :
            rn[i][j] = '0.0'
data = [pm2, pm2, pm10, o3, nox, no2, no, nmhc, co, rn]
num = len(data)
id = np.array(data, dtype = float)
id[0] = np.square(id[0])
for i in range(len(id[0])) :
    for j in range(len(id[0, 0])) :
        if (id[7, i, j] == 0) :
            id[7, i, j] = 0
        else :
            id[7, i, j] = 1 / id[7, i, j]

myans = [0]
for index in range(num) :
    out = np.dot(id[index], np.transpose(par[index]))
    myans = myans + out
myans = myans + bias

with open('linear_regression.csv', 'w') as csvfile:
    fieldnames = ['id', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for id in range(240) :
        writer.writerow({'id': 'id_'+ str(id), 'value': myans[id]})

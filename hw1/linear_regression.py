import csv
import numpy as np
import random as rand

with open('data/train.csv', 'r', encoding = 'BIG5') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader :
        data = list(reader)
    csvfile.close()

month = 0
day = 0
pm2 = [[], [], [], [], [], [], [], [], [], [], [], []]
for i in range(len(data)) :
    if ( data[i][2] == 'PM2.5' ) :
        day = day + 1
        pm2[month].extend( data[i][3:27] )
        if ( day == 20 ) :
            month = month + 1
            day = 0
        #append extend

par = []
for i in range(10) :
    par.append(rand.uniform(-1, 1))
print(par)


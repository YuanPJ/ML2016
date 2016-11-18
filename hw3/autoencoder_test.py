import pickle
import numpy as np
import sys
import csv
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2,l1l2

d5 = pickle.load(open(sys.argv[1] + 'test.p', 'rb'))
d6 = []
for i in range(10000) :
    d7 = np.reshape(d5['data'][i], (3, 32, 32))
    d6.append(d7)
xt = np.reshape(d6, (10000, 3, 32, 32))

xt = xt.astype('float32') / 255.

model = load_model(sys.argv[2])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

result = model.predict(xt, batch_size=256)
myans = np.argmax(result, axis=1)

with open(sys.argv[3], 'w') as csvfile :
    fieldnames = ['ID', 'class']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for id in range(10000) :
        writer.writerow({'ID' : id, 'class' : myans[id]})

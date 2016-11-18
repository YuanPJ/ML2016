import pickle
import numpy as np
import sys
import csv
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers import Input, BatchNormalization, Convolution2D, MaxPooling2D, UpSampling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1,l2,l1l2
from keras import backend as K

data_dir = sys.argv[1]

batch_size = 128
encoded_dim = 256
nb_classes = 10
nb_epoch = 100
startfilter = 64
add_size = 5000

d1 = pickle.load(open(data_dir + 'all_label.p', 'rb'))
d2 = np.array(d1)
xla = np.reshape(d2, (5000, 3072))

d3 = pickle.load(open(data_dir + 'all_unlabel.p', 'rb'))
d4 = np.array(d3)
xula = np.reshape(d4, (45000, 3072))

d5 = pickle.load(open(data_dir + 'test.p', 'rb'))
d6 = []
for i in range(10000) :
    d7 = np.reshape(d5['data'][i], (3, 32, 32))
    d6.append(d7)
xt = np.reshape(d6, (10000, 3072))

yla = np.zeros(5000)
for i in range(500, 5000) :
    yla[i] = int(i/500)

model = Sequential()
model.add(Dense(encoded_dim, activation='relu', input_shape=(3072,)))
model.add(Dense(encoded_dim, activation='relu'))
model.add(Dense(encoded_dim, activation='relu'))
model.add(Dense(encoded_dim, activation='relu'))
model.add(Dense(encoded_dim, activation='relu'))
model.add(Dense(encoded_dim, activation='relu'))
model.add(Dense(3072, activation='linear'))
model.summary()

model.compile(loss = 'mse', optimizer='rmsprop')
model.fit(xla, xla,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          verbose=1,
          validation_data = (xt, xt))

encoder = K.function([model.layers[0].input], [model.layers[2].output])
encoded_xla = encoder([xla])[0]

ave = np.zeros((nb_classes, encoded_dim))
for i in range(nb_classes) :
    ave[i] = np.sum(encoded_xla[500*i:500*(i+1)], axis=0)
ave = ave/500

encoded_xula = encoder([xula])[0]
c = []
for i in range(45000) :
    dif = []
    for j in range(nb_classes) :
        dif.append(np.sum((encoded_xula[i] - ave[j]) ** 2))
    label = np.argmin(dif)
    c.append((i, label, dif[label]))
c.sort(key = lambda x:x[2])

xnew = []
ynew = []
for i in range(add_size) :
    xnew.append(xula[c[i][0]])
    ynew.append(c[i][1])

xnew = np.array(xnew)
ynew = np.array(ynew)

xnew = xnew.astype('float32') / 255
xla = np.concatenate((xla, xnew), axis = 0)
yla = np.concatenate((yla, ynew), axis = 0)
xla = xla.reshape(len(xla), 3, 32, 32)

model = Sequential()
model.add(Convolution2D(startfilter, 3, 3, border_mode='same', dim_ordering='th', input_shape=(3, 32, 32)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(startfilter, 3, 3, dim_ordering='th'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), dim_ordering='th'))
model.add(Dropout(0.25))

model.add(Convolution2D(startfilter*2, 3, 3, border_mode='same', dim_ordering='th', input_shape=(3, 32, 32)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(startfilter*2, 3, 3, dim_ordering='th'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), dim_ordering='th'))
model.add(Dropout(0.25))

model.add(Convolution2D(startfilter*4, 3, 3, border_mode='same', dim_ordering='th', input_shape=(3, 32, 32)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(startfilter*4, 3, 3, dim_ordering='th'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), dim_ordering='th'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.summary()

datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

datagen.fit(xla)
model.fit_generator(datagen.flow(xla, yla,
                    batch_size=batch_size),
                    samples_per_epoch=xla.shape[0],
                    nb_epoch=nb_epoch)

model.save(sys.argv[2])

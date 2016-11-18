import pickle
import numpy as np
import sys
import csv
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1,l2,l1l2

batch_size = 256
nb_classes = 10
nb_epoch = 40
nb_semi_epoch = 5
nb_semi_data = 8192
nb_times = 100
data_augmentation = True
validation = True
nb_val = 500
startfilter = 64
filepath = sys.argv[2]

d1 = pickle.load(open(sys.argv[1] + 'all_label.p', 'rb'))
d2 = np.array(d1)
xla_total = np.reshape(d2, (5000, 3, 32, 32))

d3 = pickle.load(open(sys.argv[1] + 'all_unlabel.p', 'rb'))
d4 = np.array(d3)
xula = np.reshape(d4, (45000, 3, 32, 32))

yla_total = np.zeros(5000)
for i in range(500, 5000) :
    yla_total[i] = int(i/500)

xla = np.array([])
yla = np.array([])
xva = np.array([])
yva = np.array([])
if not validation :
    xla = xla_total
    yla = yla_total
else :
    # Spilt validation set
    split = np.random.permutation(5000)
    xla = xla_total[split[0:5000-nb_val]]
    yla = yla_total[split[0:5000-nb_val]]
    xva = xla_total[split[5000-nb_val:5000]]
    yva = yla_total[split[5000-nb_val:5000]]

xla = xla.astype('float32') / 255.
xva = xva.astype('float32') / 255.

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
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

xidx = []
yidx = []
self = False

print('Using real-time data augmentation.')

# this will do preprocessing and realtime data augmentation
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
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit_generator(datagen.flow(xla, yla,
                    batch_size=batch_size),
                    samples_per_epoch=xla.shape[0]*5,
                    callbacks = callbacks_list,
                    verbose=1,
                    validation_data=(xva, yva),
                    nb_epoch=nb_epoch)

earlyStopping = EarlyStopping(monitor='val_loss', patience=0.0, verbose=1, mode='auto')
callbacks_list = [checkpoint, earlyStopping]
for i in range(1, nb_times+1) :
    print('Times:', i, '/', nb_times)
    nb_epoch = nb_semi_epoch
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(xla, yla,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        if self :
            xtrain = np.append(xla, np.array(xula[xidx]), axis=0)
            ytrain = np.append(yla, np.array(yidx), axis=0)
            self = False
            model.fit_generator(datagen.flow(xtrain, ytrain,
                                batch_size=batch_size),
                                samples_per_epoch=xtrain.shape[0]*4,
                                nb_epoch=nb_epoch,
                                callbacks=callbacks_list,
                                verbose=1,
                                validation_data=(xva, yva))
        else :
            xtrain = xla
            ytrain = yla
            self = True
            model.fit_generator(datagen.flow(xtrain, ytrain,
                                batch_size=batch_size),
                                samples_per_epoch=xtrain.shape[0]*5,
                                nb_epoch=nb_epoch,
                                callbacks=callbacks_list,
                                verbose=1,
                                validation_data=(xva, yva))
        if self :
            gen = np.random.permutation(45000)
            rn = gen[0:nb_semi_data]
            xsemi = xula[rn]
            result = model.predict(xsemi, batch_size=512)
            high = np.argwhere(result>0.999)
            xidx = rn[high[:, 0]].tolist()
            yidx = high[:, 1].tolist()


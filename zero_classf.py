from __future__ import print_function

import sys

sys.setrecursionlimit(10000)

import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
import os.path
import time
import matplotlib.pyplot as plt

import sklearn.metrics as metrics

from keras.preprocessing.image import img_to_array
import numpy as np
import random
import cv2

batch_size = 32
nb_classes = 230
nb_epoch = 15

img_rows, img_cols = 64, 64
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 12
bottleneck = False
reduction = 0.0
dropout_rate = 0.0  # 0.0 for data augmentation

train_file_path = r'../DatasetA_train_20180813/train/'
com_path = r'../data/data_train.txt'


def load_data(dir, path):
    print("[INFO] loading images...")
    data = []
    labels = []

    data1 = []
    labels1 = []

    filelist = []
    # grab the image paths and randomly shuffle them
    with open(path, 'r') as f:
        for line in f:
            # print(line)
            filelist.append(line.split('\t'))

    # imagePaths = sorted(list(paths.list_images(path)))
    filelist = sorted(filelist)
    random.seed(42)
    random.shuffle(filelist)
    # loop over the input images
    count = 0
    for file in filelist:
        # load the image, pre-process it, and store it in the data list
        # print(file)

        imagePath = dir + file[0]
        # print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (img_rows, img_cols))
        image = img_to_array(image)
        # extract the class label from the image path and update the
        # labels list
        label = []
        label.append(file[1])
        label = np.array(label)
        if count < 32000:
            data.append(image)
            labels.append(label)
        else:
            data1.append(image)
            labels1.append(label)
        count += 1

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    data1 = np.array(data1, dtype="float") / 255.0
    labels = np.array(labels)
    labels1 = np.array(labels1)

    # convert the labels from integers to vectors
    # labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels, data1, labels1


def train():
    # model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
    #                           growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate,
    #                           bottleneck=bottleneck, reduction=reduction, weights=None)

    model = densenet.DenseNetImageNet264(input_shape=img_dim,classes = nb_classes)
    print("Model created")

    model.summary()
    optimizer = Adam(lr=1e-4)  # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    trainX, trainY, testX, testY = load_data(train_file_path, com_path)

    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    Y_train = np_utils.to_categorical(trainY, nb_classes)
    Y_test = np_utils.to_categorical(testY, nb_classes)

    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=10. / img_rows,
                                   height_shift_range=10. / img_cols)

    generator.fit(trainX, seed=0)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                   cooldown=0, patience=10, min_lr=0.5e-6)
    early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
    model_checkpoint = ModelCheckpoint("Zero.h5", monitor="val_acc", save_best_only=True,
                                       save_weights_only=True)

    callbacks = [lr_reducer, early_stopper, model_checkpoint]

    history = model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), samples_per_epoch=len(trainX),
                                  nb_epoch=nb_epoch,
                                  callbacks=callbacks,
                                  validation_data=(testX, Y_test),
                                  nb_val_samples=testX.shape[0], verbose=1)

    yPreds = model.predict(testX)
    yPred = np.argmax(yPreds, axis=1)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)

    return history


if __name__ == '__main__':
    history = train()

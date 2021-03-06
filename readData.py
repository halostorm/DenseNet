# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
import sklearn.metrics as metrics

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
# from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys

# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

sys.path.append('..')

import densenet

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 35
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 62

batch_size = 20
nb_classes = 30
nb_epoch = 300

norm_size = 64

depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0  # 0.0 for data augmentation
img_rows, img_cols = 64, 64
img_channels = 3


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-dtest", "--dataset_test", required=True,
    #                 help="path to input dataset_test")
    # ap.add_argument("-dtrain", "--dataset_train", required=True,
    #                 help="path to input dataset_train")
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


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
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        # extract the class label from the image path and update the
        # labels list
        label = file[1:31]
        label = np.array(label)
        if count < 30000:
            data.append(image)
            labels.append(label)
        else:
            data1.append(image)
            labels1.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    data1 = np.array(data1, dtype="float") / 255.0
    # labels = np.array(labels)

    # convert the labels from integers to vectors
    # labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels, data1, labels1


def train(aug, trainX, trainY, testX, testY):
    # initialize the model
    print("[INFO] compiling model...")
    img_dim = (img_channels, img_rows, img_cols)
    model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                              growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None)

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    # model.save(args["model"])

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    # plt.savefig(args["plot"])


def train1():
    batch_size = 20
    nb_classes = 30
    nb_epoch = 10

    img_rows, img_cols = 64, 64
    img_channels = 3

    img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (
        img_rows, img_cols, img_channels)
    depth = 40
    nb_dense_block = 3
    growth_rate = 12
    nb_filter = -1
    dropout_rate = 0.0  # 0.0 for data augmentation

    model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                              growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None)
    print("Model created")

    model.summary()
    optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    train_file_path = r'../DatasetA_train_20180813/train/'
    com_path = r'../data/com.txt'

    trainX, trainY, testX, testY = load_data(train_file_path, com_path)

    print(trainX.shape)
    # print(trainY.shape)
    print(testX.shape)
    # print(testY.shape)

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    trainX = densenet.preprocess_input(trainX)
    testX = densenet.preprocess_input(testX)

    Y_train = np_utils.to_categorical(trainY, nb_classes)
    Y_test = np_utils.to_categorical(testY, nb_classes)

    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5. / 32,
                                   height_shift_range=5. / 32,
                                   horizontal_flip=True)

    generator.fit(trainX, seed=0)

    # Load model
    weights_file = "weights/DenseNet-40-12-CIFAR10.h5"
    if os.path.exists(weights_file):
        # model.load_weights(weights_file, by_name=True)
        print("Model loaded.")

    out_dir = "weights/"

    lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                   cooldown=0, patience=5, min_lr=1e-5)
    model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                       save_weights_only=True, verbose=1)

    callbacks = [lr_reducer, model_checkpoint]

    model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                        steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                        callbacks=callbacks,
                        validation_data=(testX, Y_test),
                        validation_steps=testX.shape[0] // batch_size, verbose=1)

    yPreds = model.predict(testX)
    yPred = np.argmax(yPreds, axis=1)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)


if __name__ == '__main__':
    # args = args_parse()
    # train_file_path = args["dataset_train"]
    # test_file_path = args["dataset_test"]

    train1()


    # testX, testY = load_data(test_file_path,com_path)

    # construct the image generator for data augmentation
    # aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    #                          height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    #                          horizontal_flip=True, fill_mode="nearest")
    # train(aug, trainX, trainY, testX, testY)

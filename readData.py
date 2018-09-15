# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
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
    ap.add_argument("-dtest", "--dataset_test", required=True,
                    help="path to input dataset_test")
    ap.add_argument("-dtrain", "--dataset_train", required=True,
                    help="path to input dataset_train")
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
    for file in filelist:
        # load the image, pre-process it, and store it in the data list
        # print(file)
        imagePath = dir + file[0]
        # print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = file[1:31]
        label = np.array(label)
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    # labels = np.array(labels)

    # convert the labels from integers to vectors
    # labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels


def train(aug, trainX, trainY, testX, testY, args):
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
    model.save(args["model"])

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
    plt.savefig(args["plot"])


if __name__ == '__main__':
    # args = args_parse()
    # train_file_path = args["dataset_train"]
    # test_file_path = args["dataset_test"]

    train_file_path = r'../DatasetA_train_20180813/train/'
    test_file_path = r'../DatasetA_train_20180813/train/'
    com_path = r'../data/com.txt'

    trainX, trainY = load_data(train_file_path, com_path)
    # testX, testY = load_data(test_file_path,com_path)

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    # train(aug, trainX, trainY, testX, testY, args)

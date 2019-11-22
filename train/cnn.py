from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers
from keras.datasets import cifar10 #to find out more about the dataset
from skimage.io import imsave
from skimage.transform import rescale, resize
from scipy import ndimage, misc
from sklearn.model_selection import train_test_split
import numpy as np
import os
import re


class CNNModel(object):

    def __init__(self):
        self.training_images = []
        self.model = self.compile_model()
        #self.traindir2 = "/Users/rakeshkonda/Documents/CS185C/all_images/"
        self.traindir2 = "/trainingdata/"

    # Her we load images from our training images director, sort the files and load those images into an array
    def set_images(self):
        # Get images
        images = os.listdir(self.traindir2)
        images.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        filelen = len(images)-1
        print(len(images)-1)
        print("This is the length")
        # Split up training image from dataset
        self.training_images = images[:filelen]

    # Here we perform the training by taking all the images, coverting them into an numpyarray representation and loading
    # those into an array.
    # Here we also convert image from rgb to lab colorspace
    def train(self):
        xSpace = []
        ySpace = []
        for image in self.training_images:
            if(".JPEG" in image):
                train_image = img_to_array(load_img(self.traindir2+image))
                train_image = train_image / 255 # We divide by 255 to normalize the representation so its from 0-1 instead of 0-255
                xSpace.append(rgb2lab(train_image)[:, :, 0])    # this gives us the L Channel
                ySpace.append((rgb2lab(train_image)[:, :, 1:]) / 128) # We divide by 128 since ab are represented in -128 - 128
        xSpace = np.array(xSpace, dtype=float)
        ySpace = np.array(ySpace, dtype=float)
        xSpace = xSpace.reshape(len(xSpace), 64, 64, 1) # Here we reshape our array so it will fit into our model
        ySpace = ySpace.reshape(len(ySpace), 64, 64, 2)

        # Train Model. This is the function we use to perform training.
        # the variable 'history' is used for getting metrics into floydhub
        history = self.model.fit(x=xSpace, y=ySpace, validation_split=.1, batch_size=1000, epochs=100, verbose=1)

    def initialModel(self):
        # building model using keras - Floyd Model
        model = Sequential()
        model.add(InputLayer(input_shape=(64, 64, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(2, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))

        # Finish model
        model.compile(optimizer='rmsprop', loss='mse', metrics=["accuracy"])
        print("Created model!")
        return model

    def compile_model(self):
        # building model using keras - Floyd Model
        model = Sequential()
        model.add(InputLayer(input_shape=(64, 64, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        model.add(UpSampling2D((2, 2)))

        # Finish model
        model.compile(optimizer='rmsprop', loss='mse', metrics=["accuracy"])
        print("Created model!")
        return model

    def ann_model(self):
        X = []
        Y = []
        for image in os.listdir(dir):
            if '.jpg' in image:
                train_image = img_to_array(load_img(dir + image))
                #         train_image = scipy.misc.imresize(train_image, (64, 64))
                train_image = train_image / 255
                X.append(rgb2lab(train_image)[:, :, 0])
                Y.append((rgb2lab(train_image)[:, :, 1:]) / 128)

        X = np.array(X, dtype=float)  # this converts it into a huge vector
        Y = np.array(Y, dtype=float)

        X = X.reshape(len(X), 255, 255, 1)
        Y = Y.reshape(len(Y), 255, 255, 2)

        dimData_input = np.prod(X.shape[1:])
        dimData_output = np.prod(Y.shape[1:])

        X = X.reshape(X.shape[0], dimData_input)
        Y = Y.reshape(Y.shape[0], dimData_output)

        train_images, test_images, train_labels, test_labels = train_test_split(X, Y, test_size = 0.20)

        model2 = Sequential()
        model2.add(Dense(1024, activation='relu', input_shape=(input,)))
        # model4.add(Dense(1024, activation='relu'))
        # model4.add(Dense(1024, activation='relu'))
        model2.add(Dense(8192, activation='tanh'))
        model2.add(Dense(8192, activation='tanh'))
        model2.summary()

    def save_model(self):
        # Save model to disk
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model.h5")
        print("Saved model to disk")


ac = CNNModel()
ac.set_images()
ac.train()
ac.save_model()
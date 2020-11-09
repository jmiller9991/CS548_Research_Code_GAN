import keras as ks
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2 as cv

n = 230400


def imageManip(image):
    print("Code")


def buildEmotionModel(inputShape, classCnt):
    model = Sequential()

    model.add(Flatten())
    model.add(Dense(n / 1.5, activation='relu', input_shape=inputShape))
    model.add(Dense(n / 2, activation='relu'))
    model.add(Dense(n / 4, activation='relu'))
    model.add(Dense(n / 8, activation='relu'))
    model.add(Dense(classCnt, activation='relu'))

    epochs = 30
    batch_size = 128

    return model, epochs, batch_size


def main():
    inputShape = np.zeros((256, 256))
    classCnt = np.zeros((1, 6))

    model, epochs, batch_size = buildEmotionModel(inputShape, classCnt)
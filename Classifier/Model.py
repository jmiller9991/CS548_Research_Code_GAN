import keras as ks
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2 as cv
import Data.ck as ck

n = 230400

#Spit out whether an action unit is present or not
#Final output = num of action units size and 0 or 1 as present and not present

def imageManip(image):
    print("May be necessary for later but not now")


def buildEmotionModel(inputShape, classCnt):
    model = Sequential()

    model.add(Flatten())
    model.add(Dense(n / 1.5, activation='sigmoid', input_shape=inputShape))
    model.add(Dense(n / 2, activation='sigmoid'))
    model.add(Dense(n / 4, activation='sigmoid'))
    model.add(Dense(n / 8, activation='sigmoid'))
    model.add(Dense(classCnt, activation='sigmoid'))

    epochs = 30
    batch_size = 128

    return model, epochs, batch_size


def main():
    ck.main()
    allData = ck.CKDataComplete
    inputShape = np.zeros((256, 256))
    classCnt = np.zeros((1, 43))

    model, epochs, batch_size = buildEmotionModel(inputShape, classCnt)
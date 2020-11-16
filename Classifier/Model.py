import keras as ks
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2 as cv
import Data.ck as ck
import stylegan2.run_projector as proj
import stylegan2 as sg2

#Spit out whether an action unit is present or not
#Final output = num of action units size and 0 or 1 as present and not present

def imageManip(images):
    latents2 = proj.project_image_nosave(images)
    return latents2


def buildEmotionModel(inputShape, classCnt):
    model = Sequential()

    n = inputShape

    model.add(Dense(n / 1.5, activation='sigmoid', input_shape=inputShape))
    model.add(Dense(n / 2, activation='sigmoid'))
    model.add(Dropout(10))
    model.add(Dense(n / 4, activation='sigmoid'))
    model.add(Dropout(10))
    model.add(Dense(n / 8, activation='sigmoid'))
    model.add(Dropout(10))
    model.add(Dense(classCnt, activation='sigmoid'))

    epochs = 30
    batch_size = 128

    return model, epochs, batch_size

def buildConvEmotionModel(inputShape, classCnt):
    model = Sequential()

    model.add(Conv2D(10, 6, activation='relu', input_shape=inputShape))
    model.add(Conv2D(5, 6, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(2, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(300, activation='sigmoid'))
    model.add(Dense(250, activation='sigmoid'))
    model.add(Dropout(10))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dropout(10))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dropout(10))
    model.add(Dense(classCnt, activation='sigmoid'))

    epochs = 30
    batch_size = 128

    return model, epochs, batch_size

def main():
    inputShape, classCnt, subjects = ck.getLastFrameData()
    data = imageManip(inputShape)

    model_latent, epochs_latents, batch_size_latents = buildEmotionModel(data, classCnt)
    modelImage, epochsImage, batch_size_image = buildConvEmotionModel(inputShape, classCnt)
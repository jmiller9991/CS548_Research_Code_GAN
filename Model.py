import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
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

    n = (512*18)

    model.add(Flatten(input_shape=inputShape))
    model.add(Dense(n / 1.5, activation='sigmoid'))
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
    model.add(MaxPooling2D(pool_size=(2, 2)))
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
    subjects, images, emotionData, facs = ck.getLastFrameData()
    data = imageManip(images)

    model_latent, epochs_latents, batch_size_latents = buildEmotionModel(data, facs)
    modelImage, epochsImage, batch_size_image = buildConvEmotionModel(images, facs)

    history_latent = model_latent.fit(data, facs, batch_size=batch_size_latents, epochs=epochs_latents)
    history_image = modelImage.fit(images, facs, batch_size=batch_size_image, epochs=epochsImage)


if __name__ == "__main__":
    main()
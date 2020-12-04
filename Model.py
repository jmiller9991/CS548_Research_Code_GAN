import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from cv2 import cv2
import Data.ck as ck
import stylegan2.run_projector as proj
import stylegan2 as sg2
import sys
import pickle

#Spit out whether an action unit is present or not
#Final output = num of action units size and 0 or 1 as present and not present

def imageManip(images):
    latents2 = []
    for i in images:
        # Resize to 256x256
        i = cv2.resize(i, (256, 256))
        # Go from HxWxC to CxHxW
        i = i.transpose(2, 0, 1)
        # Insert a batch size of 1
        i = np.expand_dims(i, axis=0)
        # project and append to list of latents
        latents2.append(proj.project_image_nosave(sys.argv[1], i))

    # latents2 = [proj.project_image_nosave(sys.argv[1], np.expand_dims(i.transpose(2, 0, 1), axis=0)) for i in images]
    latents_array = np.asarray(latents2)
    print("Latent Array Shape: ", latents_array.shape)
    return latents_array


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
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    subjects, images, emotionData, facs = ck.getLastFrameData()
    data = imageManip(images)
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)

    model_latent, epochs_latents, batch_size_latents = buildEmotionModel(data, facs)
    modelImage, epochsImage, batch_size_image = buildConvEmotionModel(images, facs)

    history_latent = model_latent.fit(data, facs, batch_size=batch_size_latents, epochs=epochs_latents)
    history_image = modelImage.fit(images, facs, batch_size=batch_size_image, epochs=epochsImage)

    print("End")


if __name__ == "__main__":
    main()

import argparse
import numpy as np
import pickle
import sys
from typing import List

import tensorflow as tf
import tensorflow.keras as ks

from cv2 import cv2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Accuracy, Recall
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from Data import ck
from ImageBackprojector import imageManip

#Spit out whether an action unit is present or not
#Final output = num of action units size and 0 or 1 as present and not present

learning_rate = 0.01  # Default: 0.001
optimizer = ks.optimizers.Adam(learning_rate=learning_rate)
loss = "mse"
# loss = "huber_loss"
metrics = [Accuracy()]
# metrics = [Accuracy(), Recall()]


def buildEmotionModel(inputShape, classCnt):
    model = Sequential()

    n = 256

    model.add(Flatten(input_shape=inputShape[1:]))
    model.add(Dense(n // 1.5, activation='relu'))
    model.add(Dense(n // 2, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(n // 4, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(n // 8, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(classCnt, activation='sigmoid'))

    epochs = 500
    batch_size = 16

    return model, epochs, batch_size


def buildConvEmotionModel(inputShape, classCnt):
    model = Sequential()

    model.add(Conv2D(10, 6, activation='relu', input_shape=inputShape[1:]))
    model.add(Conv2D(5, 6, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(2, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(classCnt, activation='sigmoid'))

    epochs = 500
    batch_size = 128

    return model, epochs, batch_size


def get_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=argparse.FileType('rb'), required=True, help="Data pkl from ImageBackprojector.py")
    return parser.parse_args(argv)


def main():
    args = get_args(sys.argv[1:])

    # Enable multi-GPU
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    _, images, _, facs = ck.getLastFrameData((256, 256), True)

    data = pickle.load(args.data)

    # Flatten the data to a 2D array
    data = data.reshape((-1, 18 * 512))

    # Flatten the FACs into a 2D array
    facs = facs.reshape((-1, facs.shape[-1]))

    data_train, data_test, facs_train, facs_test = train_test_split(data, facs, test_size=0.2, random_state=1)

    # Number of action units
    class_count = facs.shape[-1]

    model_latent, epochs_latents, batch_size_latents = buildEmotionModel(data.shape, class_count)
    # modelImage, epochsImage, batch_size_image = buildConvEmotionModel(images.shape, class_count)

    model_latent.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history_latent = model_latent.fit(data_train, facs_train, batch_size=batch_size_latents, epochs=epochs_latents, validation_split=0.2)

    # modelImage.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # history_image = modelImage.fit(images, facs, batch_size=batch_size_image, epochs=epochsImage)

    print("End")


if __name__ == "__main__":
    main()

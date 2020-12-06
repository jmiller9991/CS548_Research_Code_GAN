import argparse
import numpy as np
import pickle
import sys
import io
from typing import List

import tensorflow as tf
import tensorflow.keras as ks

from cv2 import cv2
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from Data import ck
from ImageBackprojector import imageManip

#Spit out whether an action unit is present or not
#Final output = num of action units size and 0 or 1 as present and not present

learning_rate = 5.  # Default: 0.001
def learningRateScheduler(epoch: int) -> float:
    if (epoch % 10) == 0 and epoch != 0:
        global learning_rate
        learning_rate /= 2
    return learning_rate

optimizer = ks.optimizers.Adam(learning_rate=learning_rate)
loss = binary_crossentropy
metrics = ["binary_accuracy"]
best_model_name = "BestModel.hdf5"
callbacks = [TensorBoard(batch_size=128),
             LearningRateScheduler(learningRateScheduler, verbose=1),
             ModelCheckpoint(best_model_name, monitor='val_loss', save_best_only=True)]


def get_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=argparse.FileType('rb'), required=True, help="Train data pkl from ImageBackprojector.py")
    parser.add_argument("--test_data", type=argparse.FileType('rb'), required=True, help="Test data pkl from ImageBackprojector.py")
    return parser.parse_args(argv)


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

    epochs = 100
    batch_size = 128

    return model, epochs, batch_size


"""
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
"""


def train(latent_space: np.ndarray, images: np.ndarray, facs: np.ndarray):
    # Number of action units
    class_count = facs.shape[-1]

    model_latent, epochs_latents, batch_size_latents = buildEmotionModel(latent_space.shape, class_count)
    # modelImage, epochsImage, batch_size_image = buildConvEmotionModel(images.shape, class_count)

    model_latent.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    trained_model = model_latent.fit(latent_space, facs, batch_size=batch_size_latents, epochs=epochs_latents, callbacks=callbacks, validation_split=0.2)

    # modelImage.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # history_image = modelImage.fit(images, facs, batch_size=batch_size_image, epochs=epochsImage)


def test(latent_space: np.ndarray, images: np.ndarray, facs: np.ndarray):
    best_model = ks.models.load_model(best_model_name)
    predictions = best_model.predict(latent_space)

    # Convert to binary values
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    # Might as well use an int now
    predictions = predictions.astype(np.uint8)

    f1 = f1_score(facs, predictions, average="weighted")
    print("F1 Scores:", f1)


def main():
    args = get_args(sys.argv[1:])

    # Enable multi-GPU
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    _, images, _, facs = ck.getLastFrameData((256, 256), True)

    latent_train, facs_train = pickle.load(args.train_data)
    latent_test, facs_test = pickle.load(args.test_data)
    # Flatten the data to a 2D array
    latent_train = latent_train.reshape((-1, 18 * 512))
    latent_test = latent_test.reshape((-1, 18 * 512))

    # Flatten the FACs into a 2D array
    facs_train = facs_train.reshape((-1, facs_train.shape[-1]))
    facs_test = facs_test.reshape((-1, facs_test.shape[-1]))

    train(latent_train, images, facs_train)
    test(latent_test, images, facs_test)


if __name__ == "__main__":
    main()

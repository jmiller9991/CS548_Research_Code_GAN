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
callbacks = [TensorBoard(batch_size=128),
             LearningRateScheduler(learningRateScheduler, verbose=1),
             ModelCheckpoint("BestModel.hdf5", monitor='val_loss', save_best_only=True)]


def get_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=argparse.FileType('rb'), required=True, help="Data pkl from ImageBackprojector.py")

    subparsers = parser.add_subparsers(help="Sub-commands", dest="command")
    test = subparsers.add_parser("test")
    test.add_argument("-m", "--model", type=str, required=True, help="Saved model from RunModel.py")
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

    epochs = 300
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
    latent_space_train, latent_space_test, facs_train, facs_test = train_test_split(latent_space, facs, test_size=0.2, random_state=1)

    # Number of action units
    class_count = facs.shape[-1]

    model_latent, epochs_latents, batch_size_latents = buildEmotionModel(latent_space.shape, class_count)
    # modelImage, epochsImage, batch_size_image = buildConvEmotionModel(images.shape, class_count)

    model_latent.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history_latent = model_latent.fit(latent_space_train, facs_train, batch_size=batch_size_latents, epochs=epochs_latents, callbacks=callbacks, validation_split=0.2)

    # modelImage.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # history_image = modelImage.fit(images, facs, batch_size=batch_size_image, epochs=epochsImage)


def test(model_path: str, latent_space: np.ndarray, images: np.ndarray, facs: np.ndarray):
    model = ks.models.load_model(model_path)

    predictions = model.predict(latent_space)
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
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

    if(facs.dtype != np.uint8):
        facs = facs.astype(np.uint8)

    data = pickle.load(args.data)
    # Flatten the data to a 2D array
    data = data.reshape((-1, 18 * 512))

    # Flatten the FACs into a 2D array
    facs = facs.reshape((-1, facs.shape[-1]))

    if args.command == "test":
        test(args.model, data, images, facs)
    else:
        train(data, images, facs)

    print("End")


if __name__ == "__main__":
    main()

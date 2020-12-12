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

learning_rate = 0.001
def learningRateScheduler(epoch: int, interval: int = 10) -> float:
    if (epoch % interval) == 0 and epoch != 0:
        global learning_rate
        learning_rate /= 2
    return learning_rate

optimizer = ks.optimizers.Adam(learning_rate=learning_rate)
loss = binary_crossentropy
metrics = ["binary_accuracy"]
best_model_name = "BestModel.hdf5"
callbacks = [TensorBoard(batch_size=128),
             ModelCheckpoint(best_model_name, monitor='val_loss', save_best_only=True)]

def get_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=argparse.FileType('rb'), required=True, help="Train data pkl from ImageBackprojector.py")
    parser.add_argument("--test_data", type=argparse.FileType('rb'), required=True, help="Test data pkl from ImageBackprojector.py")
    parser.add_argument("-p", "--percent_active", type=float, default=0, help="Minimum percentage of activation of AUs as a decimal")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Epochs to train for")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("-l", "--lr_interval", type=int, default=10, help="Number of epochs between learning rate halving")
    parser.add_argument("-t", "--test", action="store_true", default=False, help="Test the model after training has completed")
    parser.add_argument("-c", "--ck", required=True, help="Path to CK Data")
    return parser.parse_args(argv)


def buildEmotionModel(inputShape, classCnt):
    model = Sequential()

    n = 4096  # Originally 256

    print(inputShape[1:])
    model.add(Flatten(input_shape=inputShape[1:]))
    model.add(Dense(n // 1.5, activation='relu'))
    model.add(Dense(n // 2, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(n // 4, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(n // 8, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(classCnt, activation='sigmoid'))

    return model


def train(latent_space: np.ndarray, images: np.ndarray, facs: np.ndarray, epochs: int, batch_size: int, lr_interval: int):
    # Number of action units
    class_count = facs.shape[-1]

    model_latent = buildEmotionModel(latent_space.shape, class_count)

    model_latent.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model_latent.fit(latent_space, facs, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_split=0.2)


def test(latent_space: np.ndarray, images: np.ndarray, facs: np.ndarray):
    best_model = ks.models.load_model(best_model_name)
    predictions = best_model.predict(latent_space)

    # Convert to binary values
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    # Might as well use an int now
    predictions = predictions.astype(np.uint8)

    for fac in range(facs.shape[-1]):
        print(f"FAC at index {fac}")
        f1 = f1_score(facs[:, fac], predictions[:, fac])
        print("F1 Score:", f1)

        con_mat = confusion_matrix(predictions[:, fac], facs[:, fac])
        print("Confusion Matrix:\n", con_mat)


def main():
    args = get_args(sys.argv[1:])

    # Enable multi-GPU
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    _, images, _, facs, au_sum = ck.getLastFrameData(args.ck, (256, 256), True)
    _, indices_of_interest = ck.get_aus_with_n_pct_positive(au_sum, args.percent_active)

    # Load the data
    latent_train, facs_train = pickle.load(args.train_data)
    latent_test, facs_test = pickle.load(args.test_data)

    # Flatten the data to a 2D array
    latent_train = latent_train.reshape((-1, 18 * 512))
    latent_test = latent_test.reshape((-1, 18 * 512))

    # Only keep the AUs that meet the % active requirement
    facs_train = facs_train[:, :, indices_of_interest]
    facs_test = facs_test[:, :, indices_of_interest]

    # Flatten the FACs into a 2D array
    facs_train = facs_train.reshape((-1, facs_train.shape[-1]))
    facs_test = facs_test.reshape((-1, facs_test.shape[-1]))

    # Add the learning rate scheduler
    global callbacks
    callbacks.append(LearningRateScheduler(lambda epoch:
                                           learningRateScheduler(epoch,
                                                                 interval=args.lr_interval),
                                           verbose=1))

    train(latent_train, images, facs_train, args.epochs, args.batch_size, args.lr_interval)

    if(args.test):
        test(latent_test, images, facs_test)


if __name__ == "__main__":
    main()

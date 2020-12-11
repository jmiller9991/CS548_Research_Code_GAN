#!/opt/anaconda3/bin/python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.inptus import Input

from Data import ck
from stylegan2 import run_projector

EPOCHS = 30
BATCH_SIZE = 128

def imageManip(images):
    latents = run_projector.project_image_nosave(images)
    return latents

def build_model(latent_shape: tuple, classCount: int):
    latent_size = np.prod(latent_shape)

    input_layer = Input(shape=latent_shape)

    emotion_dense1 = Dense(latent_size / 1.5, activation='sigmoid')(input_layer)
    emotion_dense2 = Dense(latent_size / 2, activation='sigmoid')(emotion_dense1)
    emotion_dropout1 = Dropout(10)(emotion_dense2)
    emotion_dense3 = Dense(latent_size / 4, activation='sigmoid')(emotion_dropout1)
    emotion_dropout2 = Dropout(10)(emotion_dense3)
    emotion_dense4 = Dense(latent_size / 8, activation='sigmoid')(emotion_dropout2)
    emotion_dropout3 = Dropout(10)(emotion_dense4)
    emotion_dense5 = Dense(classCount, activation='sigmoid')(emotion_dropout3)
    au_vector = Dense(classCount, activation="sigmoind")(emotion_dense5)

    latent_conv1 = Conv2D(10, 6, activation='relu')(input_layer)
    latent_conv2 = Conv2D(5, 6, activation='relu')(latent_conv1)
    latent_pool1 = MaxPooling2D(pool_size=(2, 2))(latent_conv2)
    latent_conv3 = Conv2D(3, 3, activation='relu')(latent_pool1)
    latent_pool2 = MaxPooling2D(pool_size=(2, 2))(latent_conv3)
    latent_conv4 = Conv2D(2, 3, activation='relu')(latent_pool2)
    latent_pool3 = MaxPooling2D(pool_size=(2, 2))(latent_conv4)
    latent_flatten = Flatten()(latent_pool3)
    latent_dense1 = Dense(300, activation='sigmoid')(latent_flatten)
    latent_dense2 = Dense(250, activation='sigmoid')(latent_dense1)
    latent_dropout1 = Dropout(10)(latent_dense2)
    latent_dense3 = Dense(200, activation='sigmoid')(latent_dropout1)
    latent_dropout2 = Dropout(10)(latent_dense3)
    latent_dense4 = Dense(100, activation='sigmoid')(latent_dropout2)
    latent_dropout3 = Dropout(10)(latent_dense4)
    latent_representation = Dense(classCount, activation='sigmoid')(latent_dropout3)

    return keras.Model(inputs=input_layer, outputs=[latent_representation, au_vector])


def main():
    inputShape, classCnt, subjects = ck.getLastFrameData()
    data = imageManip(inputShape)

    model = build_model(data.shape, classCnt)


if __name__ == "__main__":
    main()


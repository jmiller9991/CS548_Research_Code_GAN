import keras as ks
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def buildModel():
    model = Sequential()

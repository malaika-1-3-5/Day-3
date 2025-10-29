import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

# to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Architecture
model = Sequential()
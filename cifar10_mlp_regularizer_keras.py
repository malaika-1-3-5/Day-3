import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

# to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Architecture
model_base = Sequential()
model_base.add(Flatten(input_shape=(32, 32, 3)))
model_base.add(Dense(1024, activation='relu'))
model_base.add(Dense(512, activation='relu'))
model_base.add(Dense(256, activation='relu'))
model_base.add(Dense(128, activation='relu'))
model_base.add(Dense(64, activation='relu'))
model_base.add(Dense(100, activation='softmax')) #bcoz we have 100 classes

# Compile
model_base.compile(optimizer = Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train
result_1 = model_base.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate
test_loss, test_accuracy = model_base.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")
print(result_1.history.keys())
print(result_1.history.values())
print(result_1.history)

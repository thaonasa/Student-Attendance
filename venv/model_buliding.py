from tensorflow import keras
import numpy as np
import os
from PIL import Image

TRAIN_DATA = 'datasets/train-data'
TEST_DATA = 'datasets/test-data'


Xtrain = []
Ytrain = []

Xtrain = [()]

Xtest = []
Ytest = []

dict = {'folder train path': [1, 0, 0, 0, 0],
        'folder train path': [0, 1, 0, 0, 0],
        'folder train path': [0, 0, 1, 0, 0],
        'folder train path': [0, 0, 0, 1, 0],
        'folder train path': [0, 0, 0, 0, 1],
        'folder test path': [1, 0, 0, 0, 0],
        'folder test path': [0, 1, 0, 0, 0],
        'folder test path': [0, 0, 1, 0, 0],
        'folder test path': [0, 0, 0, 1, 0],
        'folder test path': [0, 0, 0, 0, 1]}


def getData(dirData, lstData):
    for whatever in os.listdir(dirData):
        whatever_path = os.path.join(dirData, whatever)
        lst_filename_path = []
        for filename in os.listdir(whatever_path):
            filename_path = os.path.join(whatever_path, filename)
            label = filename_path.split('\\')[1]

            print(filename_path)
            img = np.array(Image.open(filename_path))
            lst_filename_path.append(img, dict[label])

        lstData.extend(lst_filename_path)
    return lstData


Xtrain = getData(TRAIN_DATA, Xtrain)
Xtest = getData(TEST_DATA, Xtest)

models = keras.models
layers = keras.layers

model_training_first = models.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Droupout(0.15),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Droupout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Droupout(0.2),

    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(5, activation='softmax'),
])

model_training_first.summary()

model_training_first.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

model_training_first.fit(Xtrain, ytrain_ohc, epochs=10)

model_training_first.save('model-cifar10_10epochs.h5')

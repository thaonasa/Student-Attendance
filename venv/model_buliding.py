import cv2
from tensorflow import keras
import numpy as np
import os
from PIL import Image

TRAIN_DATA = 'datasets/train-data'
TEST_DATA = 'datasets/test-data'


Xtrain = []
Ytrain = []

# Xtrain = [(mth1, ohe1), (mth2, ohe2), ..........., (mthn), ohen]
# Xtrain[0][0], Xtrain[0][1]
# Xtrain = [x[0] for i, x in enumerate(Xtrain)]


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

for i in 10:
    np.random.shuffle(Xtrain)  # xáo trộn dữ liệu

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

model_training_first.fit(np.array([x[0] for i, x in enumerate(Xtrain)]), np.array(
    [y[1] for _, y in enumerate(Xtrain)]), epochs=10)

model_training_first.save('model-cifar10_10epochs.h5')

models = models.load_model('model-cifar10_50epochs.h5')

lstResult = ['title1', 'title2', 'title3', 'titlen']
face_detector = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_alt.xml')
cam = cv2.VideoCapture()
while True:
    OK, frame = cam.read()
    faces = face_detector.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = cv2.resize(frame[y: y+h, x: x+w], (128, 128))
        result = np.argmax(models.predict(roi.reshape((-1, 128, 128, 3))))
        # roi = cv2.resize(frame[y+2: y+h-2, x+2: x+w-2], (100, 100))
        # cv2.imwrite('imgs_roi/roi_{}.jpg'.format(count), roi)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 50), 1)
        # count += 1
        cv2.putText(frame, lstResult(result), (x+15, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('FRAME', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

# train_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_facenet_model():
    """Build and return a simple CNN model for FaceNet."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # 128 output classes (adjust as needed)
    model.add(Dense(128, activation='softmax'))

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model():
    """Train the FaceNet model using augmented data."""
    model = build_facenet_model()

    # Augment training data
    datagen = ImageDataGenerator(rescale=1./255)
    train_data = datagen.flow_from_directory(
        '/data/Original Images', target_size=(160, 160), batch_size=32, class_mode='categorical')

    # Train the model
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_data, epochs=10, validation_data=train_data,
              callbacks=[early_stopping])

    # Save the trained model
    model.save('/models/facenet_model/facenet.h5')

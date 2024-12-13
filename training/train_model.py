import os
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# Đường dẫn dữ liệu và mô hình
TRAIN_DIR = "data/Face Mask Dataset/Train"
VALIDATION_DIR = "data/Face Mask Dataset/Validation"
TEST_DIR = "data/Face Mask Dataset/Test"
MODEL_SAVE_PATH = "models/mask_model/mask_model.h5"


def build_model(num_classes):
    """
    Build and return a MobileNetV2 model for classification.
    """
    base_model = MobileNetV2(
        weights='imagenet', include_top=False, input_shape=(160, 160, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout to reduce overfitting
    predictions = Dense(num_classes, activation='softmax')(
        x)  # Number of classes in output

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model():
    """
    Train the classification model using the prepared dataset.
    """
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(160, 160),
        batch_size=32,
        class_mode='categorical'
    )

    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(160, 160),
        batch_size=32,
        class_mode='categorical'
    )

    # Load test data
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(160, 160),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # Shuffle is False to maintain the order for evaluation
    )

    # Get the number of classes from the training data
    num_classes = train_generator.num_classes
    print(f"Number of classes: {num_classes}")

    # Build the model
    model = build_model(num_classes)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1)

    # Train the model
    print("Starting training...")
    model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    print(f"Model training completed. Model saved to: {MODEL_SAVE_PATH}")

    # Evaluate the model on the test set
    evaluate_model(model, test_generator)


def evaluate_model(model, test_generator):
    """
    Evaluate the trained model on the test set and calculate additional metrics.
    """
    print("Evaluating model on test data...")

    # Get true labels and predictions
    y_true = test_generator.classes
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Get class labels
    class_labels = list(test_generator.class_indices.keys())

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    # Classification report
    report = classification_report(
        y_true, y_pred_classes, target_names=class_labels)
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")

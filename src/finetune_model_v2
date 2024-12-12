import os
from keras.models import load_model, Model
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Đường dẫn dữ liệu và mô hình
TRAIN_DIR = "data/Original Images/Train"
VALIDATION_DIR = "data/Original Images/Validation"
TEST_DIR = "data/Original Images/Test"
PRETRAINED_MODEL_PATH = "models/mask_model/mask_model.h5"
FINAL_MODEL_SAVE_PATH = "models/final_model/final_model.h5"


def fine_tune_model(pretrained_model_path, num_classes):
    """
    Load a pretrained model and modify it for fine-tuning with the new number of classes.
    """
    # Load the pretrained model
    model = load_model(pretrained_model_path)

    # Remove the last layer (output layer) and add custom layers
    x = model.layers[-2].output  # Get output of the second-to-last layer
    x = Dense(256, activation='relu', name="custom_dense_1")(x)
    x = Dropout(0.5, name="custom_dropout_1")(
        x)  # Dropout to prevent overfitting
    predictions = Dense(num_classes, activation='softmax',
                        name="custom_output")(x)

    # Create the new model
    new_model = Model(inputs=model.input, outputs=predictions)

    # Unfreeze all layers for fine-tuning
    for layer in new_model.layers:
        layer.trainable = True

    # Compile the model with a lower learning rate
    new_model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    return new_model


def train_finetune_model():
    """
    Fine-tune the pretrained model on the new dataset.
    """
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
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
        shuffle=False  # Ensure consistency for evaluation
    )

    # Get the number of classes from the training data
    num_classes = train_generator.num_classes
    print(f"Number of classes: {num_classes}")

    # Load and fine-tune the model
    model = fine_tune_model(PRETRAINED_MODEL_PATH, num_classes)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(
        filepath=FINAL_MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

    # Train the model
    print("Starting fine-tuning...")
    model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )

    print(f"Fine-tuning completed. Model saved to: {FINAL_MODEL_SAVE_PATH}")

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
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    report = classification_report(
        y_true, y_pred_classes, target_names=class_labels)
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    try:
        train_finetune_model()
    except Exception as e:
        print(f"Error during fine-tuning: {e}")

# data_augmentation.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def augment_image(image):
    """Apply data augmentation techniques on an image."""
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Reshape to (1, width, height, channels)
    image = image.reshape((1,) + image.shape)
    augmented_images = datagen.flow(image)
    return augmented_images

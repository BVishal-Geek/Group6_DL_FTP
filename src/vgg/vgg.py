import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, backend as K
from sklearn.metrics import accuracy_score, f1_score

# Base directories for common data and Excel files
common_base_dir = '/home/ubuntu/Final_Project/data/'
excel_dir = '/home/ubuntu/Final_Project/excel/'

# Image size for model input
IMG_SIZE = (224, 224)

# Batch size for data loading
BATCH_SIZE = 32


def create_data_generator(excel_file, image_folder):
    """Create a data generator for loading images and labels."""
    df = pd.read_excel(excel_file)
    datagen = ImageDataGenerator(rescale=1. / 255)  # Normalize pixel values to [0, 1]

    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_folder,
        x_col='imageid',
        y_col='classification',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='raw',  # Use 'raw' for regression or binary classification without one-hot encoding
        shuffle=True
    )

    return generator


# Create data generators for train, validation, and test datasets
train_generator = create_data_generator(
    os.path.join(excel_dir, 'train.xlsx'),
    os.path.join(common_base_dir, 'common_train')
)
val_generator = create_data_generator(
    os.path.join(excel_dir, 'validation.xlsx'),
    os.path.join(common_base_dir, 'common_validation')
)
test_generator = create_data_generator(
    os.path.join(excel_dir, 'test.xlsx'),
    os.path.join(common_base_dir, 'common_test')
)


# Define the VGG-like architecture from scratch
def build_vgg_model(input_shape):
    model = models.Sequential()

    # VGG-style convolutional blocks
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))

    # Output layer with sigmoid for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


# Define Focal Loss function
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weights = alpha * y_true + (1 - alpha) * (1 - y_true)
        focal_loss = weights * K.pow((1 - y_pred), gamma) * cross_entropy
        return K.mean(focal_loss)

    return focal_loss_fixed


# Build the VGG-like model
model = build_vgg_model((IMG_SIZE[0], IMG_SIZE[1], 3))

# Compile the VGG-like model with Focal Loss
model.compile(optimizer=optimizers.Adam(),
              loss=focal_loss(gamma=2.0, alpha=0.25),
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model using the data generators
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
)

# Evaluate the model on test data using the test generator
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)[:, 0]

# Calculate accuracy and F1 score on test data
test_labels = np.concatenate([test_generator[i][1] for i in range(len(test_generator))])
accuracy = accuracy_score(test_labels, predicted_classes)
f1 = f1_score(test_labels, predicted_classes)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import backend as K

# Paths to Excel files and image directories
excel_dir = '/home/ubuntu/Final_Project/excel/'
image_dir = '/home/ubuntu/Final_Project/data/faceapp/'

# Image size for VGG input
IMG_SIZE = (224, 224)


# Function to load data from Excel files with '_mask' in their names
def load_mask_data(folder_name):
    excel_path = os.path.join(excel_dir, f"{folder_name}.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Load the Excel file
    df = pd.read_excel(excel_path)

    # Prepare image paths and labels
    images = []
    labels = []
    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, folder_name, row['imageid'])
        if os.path.exists(image_path):
            img = load_img(image_path, target_size=IMG_SIZE)  # Load and resize image
            img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array)
            labels.append(row['classification'])

    return np.array(images), np.array(labels)


# Load train, validation, and test data from masked Excel files
train_images, train_labels = load_mask_data('train')
val_images, val_labels = load_mask_data('validation')
test_images, test_labels = load_mask_data('test')


# Define the VGG-like architecture from scratch
def build_vgg_model(input_shape):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
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

# Data augmentation for training data to prevent overfitting
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)
datagen.fit(train_images)

# Train the model
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    validation_data=(val_images, val_labels),
    epochs=10,
)

# Evaluate the model on test data
predictions = model.predict(test_images)
predicted_classes = (predictions > 0.5).astype(int)[:, 0]

# Calculate accuracy and F1 score
accuracy = accuracy_score(test_labels, predicted_classes)
f1 = f1_score(test_labels, predicted_classes)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
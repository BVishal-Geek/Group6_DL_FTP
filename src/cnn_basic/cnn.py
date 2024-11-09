import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Paths to Excel files and image directories
excel_dir = '/home/ubuntu/Final_Project/excel/'
image_dir = '/home/ubuntu/Final_Project/data/faceapp/'

# Image size for resizing
IMG_SIZE = (128, 128)


# Function to load data from Excel and prepare it
def load_data(folder_name):
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


# Load train, validation, and test data
train_images, train_labels = load_data('train')
val_images, val_labels = load_data('validation')
test_images, test_labels = load_data('test')

# Define the CNN architecture
model = models.Sequential()

# Layer 1
model.add(layers.Conv2D(50, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Layer 2
model.add(layers.Conv2D(100, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Layer 3
model.add(layers.Conv2D(150, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Layer 4
model.add(layers.Conv2D(100, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Layer 5
model.add(layers.Conv2D(50, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Data augmentation for training data (optional)
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
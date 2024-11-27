#%%
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import os

#%%
# Parameters
real_dir = '/home/ubuntu/Group6_DL_FTP/data/images/real/'
fake_dir = '/home/ubuntu/Group6_DL_FTP/data/images/fake/'
batch_size = 32
img_height = 128
img_width = 128
epochs = 10
max_images_per_class = 200000  # Limit the number of images from each class

# Load and preprocess images
def load_data_from_directory(directory, label, img_height, img_width, max_images=None):
    images = []
    labels = []
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Get a sorted list of files and limit to `max_images`
    img_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if max_images is not None:
        img_files = img_files[:max_images]

    for img_file in img_files:
        img_path = os.path.join(directory, img_file)
        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(label)

    print(f"Loaded {len(images)} images from {directory}")
    return np.array(images), np.array(labels)

# Load real and fake images, limiting to 200,000 each
real_images, real_labels = load_data_from_directory(real_dir, label=0, img_height=img_height, img_width=img_width, max_images=max_images_per_class)
fake_images, fake_labels = load_data_from_directory(fake_dir, label=1, img_height=img_height, img_width=img_width, max_images=max_images_per_class)

# Combine and shuffle data
images = np.concatenate([real_images, fake_images], axis=0) / 255.0
labels = np.concatenate([real_labels, fake_labels], axis=0)

indices = np.arange(images.shape[0])
np.random.shuffle(indices)
images, labels = images[indices], labels[indices]

# Split into training and validation sets (80% train, 20% validation)
split_idx = int(0.8 * len(images))
train_images, val_images = images[:split_idx], images[split_idx:]
train_labels, val_labels = labels[:split_idx], labels[split_idx:]

print(f"Training set size: {len(train_images)}")
print(f"Validation set size: {len(val_images)}")

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Define CNN model
model = models.Sequential([
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Predict and evaluate metrics
val_predictions = (model.predict(val_ds.map(lambda x, y: x)).ravel() > 0.5).astype("int32")
val_labels_flat = tf.concat([y for _, y in val_ds], axis=0).numpy()

print("\nClassification Report:")
print(classification_report(val_labels_flat, val_predictions, target_names=["Real", "Fake"]))

print("\nConfusion Matrix:")
print(confusion_matrix(val_labels_flat, val_predictions))

val_probabilities = model.predict(val_ds.map(lambda x, y: x)).ravel()
auc_roc = roc_auc_score(val_labels_flat, val_probabilities)
print(f"\nAUC-ROC Score: {auc_roc:.4f}")
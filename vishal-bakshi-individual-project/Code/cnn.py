import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np


# Parameters
batch_size = 2
img_height = 64
img_width = 64
epochs = 10

# Load data lazily using image_dataset_from_directory
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    '/home/ubuntu/Group6_DL_FTP/data/images',  # Parent directory containing 'real' and 'fake'
    validation_split=0.2,  # Split 80% train, 20% validation
    subset="both",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch data for efficient GPU utilization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(3)
val_ds = val_ds.prefetch(3)

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

# Generate predictions for validation set
val_predictions = (model.predict(val_ds.map(lambda x, y: x)).ravel() > 0.5).astype("int32")
val_labels_flat = tf.concat([y for _, y in val_ds], axis=0).numpy()

# Classification metrics
print("\nClassification Report:")
print(classification_report(val_labels_flat, val_predictions, target_names=["Fake", "Real"]))

print("\nConfusion Matrix:")
print(confusion_matrix(val_labels_flat, val_predictions))

# AUC-ROC score
val_probabilities = model.predict(val_ds.map(lambda x, y: x)).ravel()
auc_roc = roc_auc_score(val_labels_flat, val_probabilities)
print(f"\nAUC-ROC Score: {auc_roc:.4f}")
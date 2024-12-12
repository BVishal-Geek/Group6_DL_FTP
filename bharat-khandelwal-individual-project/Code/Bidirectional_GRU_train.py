import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import tensorflow.keras.backend as K
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy("mixed_float16")


# Constants
FRAME_SIZE = (299, 299)  # InceptionV3 expects 299x299 images
NUM_FEATURES = 2048  # Features extracted from InceptionV3's pooling layer
MAX_SEQ_LENGTH = 16
BATCH_SIZE = 8  # Batch size for training

# Load Pretrained Feature Extractor (InceptionV3)
feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")


def crop_center_square(frame):
    """Crop the center square of the frame."""
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


# Importing Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def crop_face(frame):
    """Detect and crop the face from the frame."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        # Take the first detected face
        x, y, w, h = faces[0]
        face = frame[y:y + h, x:x + w]
        return cv2.resize(face, FRAME_SIZE)  # Resize to match input size
    else:
        # Fallback to center crop if no face is detected
        return crop_center_square(frame)


def extract_frame_features(video_path, max_frames=MAX_SEQ_LENGTH):
    """Extract features for a fixed number of frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            # Detect and crop face, or fallback to center crop
            try:
                frame = crop_face(frame)
                frame = cv2.resize(frame, FRAME_SIZE)  # Ensure consistent dimensions
                frame = preprocess_input(frame)  # Normalize for InceptionV3
                frames.append(frame)
            except Exception as e:
                print(f"Error processing frame in {video_path}: {e}")
                continue

    cap.release()

    # Ensure consistent frame count
    if len(frames) < max_frames:
        print(f"Skipping video {video_path} due to insufficient frames ({len(frames)}/{max_frames}).")
        return None  # Skip videos with fewer than max_frames

    # Batch process frames with feature extractor
    frames = np.array(frames)
    features = feature_extractor.predict(frames, verbose=0)

    return features


def video_data_generator(video_paths, labels, batch_size, max_frames=MAX_SEQ_LENGTH):
    """Generator for video data."""
    while True:
        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            X_batch = []
            y_batch = []

            for video_path, label in zip(batch_paths, batch_labels):
                features = extract_frame_features(video_path, max_frames=max_frames)
                if features is not None:
                    X_batch.append(features)
                    y_batch.append(label)

            yield np.array(X_batch), np.array(y_batch)


def split_videos(real_paths, fake_paths, test_size=0.3, val_size=0.5, random_state=42):
    """Split videos into train, validation, and test sets using stratified sampling."""
    # Load video paths
    real_videos = [os.path.join(real_paths[0], f) for f in os.listdir(real_paths[0]) if f.endswith(('.mp4', '.avi', '.mkv'))]
    fake_videos = [os.path.join(fake_paths[0], f) for f in os.listdir(fake_paths[0]) if f.endswith(('.mp4', '.avi', '.mkv'))]

    print(f"Number of real videos: {len(real_videos)}")
    print(f"Number of fake videos: {len(fake_videos)}")

    # Assign labels
    real_labels = [0] * len(real_videos)
    fake_labels = [1] * len(fake_videos)

    # Combine data
    all_videos = real_videos + fake_videos
    all_labels = real_labels + fake_labels

    # Stratify the first split
    train_videos, temp_videos, train_labels, temp_labels = train_test_split(
        all_videos, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )

    # Stratify the second split (validation vs. test)
    val_videos, test_videos, val_labels, test_labels = train_test_split(
        temp_videos, temp_labels, test_size=val_size, random_state=random_state, stratify=temp_labels
    )

    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = -alpha_t * K.pow(1. - p_t, gamma) * K.log(p_t)
        return K.mean(focal_loss, axis=-1)
    return focal_loss_fixed

def build_bidirectional_gru_attention_model(input_shape=(MAX_SEQ_LENGTH, NUM_FEATURES)):
    """Build a Bidirectional GRU model with Temporal Attention."""
    # Frame features input
    frame_features_input = tf.keras.Input(shape=input_shape, name="frame_features")

    # Bidirectional GRUs
    x = layers.Bidirectional(layers.GRU(32, return_sequences=True))(frame_features_input)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.GRU(16, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)

    # Attention mechanism
    attention_weights = layers.Dense(1, activation="tanh")(x)
    attention_weights = layers.Flatten()(attention_weights)
    attention_weights = layers.Activation("softmax", name="attention_weights")(attention_weights)
    attention_weights = layers.RepeatVector(32)(attention_weights)
    attention_weights = layers.Permute([2, 1])(attention_weights)

    # Apply attention to sequence
    x = layers.Multiply()([x, attention_weights])
    x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)

    # Fully connected layers
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    # Build and compile the model
    model = models.Model(inputs=frame_features_input, outputs=output)
    model.compile(
        loss=focal_loss(gamma=2., alpha=0.866),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"]
    )
    return model

#%%
# Main Script
real_paths = ["/home/ubuntu/DL_Project/archive/all_real/"]
fake_paths = ["/home/ubuntu/DL_Project/archive/Celeb-synthesis/"]

# Step 1: Split videos into train, validation, and test sets
(train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = split_videos(
    real_paths, fake_paths
)

# Step 2: Create data generators
train_gen = video_data_generator(train_videos, train_labels, batch_size=BATCH_SIZE)
val_gen = video_data_generator(val_videos, val_labels, batch_size=BATCH_SIZE)

# Step 3: Build and train the model
model = build_bidirectional_gru_attention_model()
print("Training the Bidirectional GRU + Attention model...")


history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_videos) // BATCH_SIZE,
    validation_steps=len(val_videos) // BATCH_SIZE,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model_bidirectional.keras", monitor="val_loss", save_best_only=True)
    ]
)

print(model.summary())
#%%
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import sys

# Redirect standard output to a log file
log_file = open("evaluation_log.txt", "w")
sys.stdout = log_file

# Step 4: Evaluate the model on the test data generator
test_gen = video_data_generator(test_videos, test_labels, batch_size=BATCH_SIZE)

print("Evaluating the model...")
test_steps = len(test_videos) // BATCH_SIZE
test_loss, test_accuracy = model.evaluate(test_gen, steps=test_steps)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 5: Generate predictions and calculate metrics
predictions = model.predict(test_gen, steps=test_steps)
y_pred = (predictions > 0.5).astype(int).flatten()
y_test = np.array(test_labels[:len(y_pred)])  # Adjust length to match predictions

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# F1 Macro Score
f1_macro = f1_score(y_test, y_pred, average="macro")
print(f"\nF1 Macro Score: {f1_macro:.4f}")
# Close the log file
sys.stdout = sys.__stdout__  # Restore standard output
log_file.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_plot_file = "confusion_matrix.png"
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(cm_plot_file)
plt.show()

# Step 6: Plot Training and Validation Loss Curves
def plot_training_curves(history,filename="training_curves.png" ):
    """Plot the training and validation loss and accuracy curves."""
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(filename)
    plt.close()

# Plot the curves
plot_training_curves(history)

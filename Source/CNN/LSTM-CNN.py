import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# Constants
FRAME_SIZE = (224, 224)  # ResNet50 expects 224x224 images
NUM_FEATURES = 512  # Features extracted from ResNet50
BATCH_SIZE = 4  # Batch size for LSTM
MAX_FRAMES = None  # No frame limit, process the entire video

# Pretrained ResNet50 for feature extraction
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
feature_extractor = models.Model(inputs=base_model.input, outputs=base_model.output)


def is_video_valid(video_path):
    """
    Check if a video file is valid by attempting to open and read the first frame.
    Args:
        video_path (str): Path to the video file.
    Returns:
        bool: True if the video is valid, False otherwise.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return False
    ret, _ = cap.read()
    cap.release()
    return ret


def preprocess_video(video_path):
    """
    Process an entire video, frame by frame, for face detection and preprocessing.
    Args:
        video_path (str): Path to the video file.
    Returns:
        np.ndarray: Processed frames, each as a feature vector.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_input(cv2.resize(frame, FRAME_SIZE))
        frames.append(frame)

    cap.release()
    if not frames:
        return None
    return feature_extractor.predict(np.array(frames), batch_size=1, verbose=0)


def process_videos(video_paths, labels):
    """
    Extract features from all videos in the dataset.
    Args:
        video_paths (list): List of video paths.
        labels (list): List of corresponding labels.
    Returns:
        list, list: Feature arrays (X) and labels (y).
    """
    X, y = [], []
    for video_path, label in zip(video_paths, labels):
        if not is_video_valid(video_path):
            print(f"Skipping corrupted video: {video_path}")
            continue
        features = preprocess_video(video_path)
        if features is not None:
            X.append(features)
            y.append(label)
    return X, np.array(y)


def pad_features(features, max_frames=None):
    """
    Pad or truncate features to a fixed length for LSTM input.
    Args:
        features (list): List of feature arrays, one per video.
        max_frames (int): Maximum number of frames to pad or truncate to.
    Returns:
        np.ndarray: Padded or truncated feature array.
    """
    max_len = max(len(f) for f in features) if max_frames is None else max_frames
    padded_features = []
    for f in features:
        if len(f) > max_len:
            padded_features.append(f[:max_len])
        else:
            pad_width = max_len - len(f)
            padded_features.append(np.pad(f, ((0, pad_width), (0, 0)), mode="constant"))
    return np.array(padded_features)


def build_lstm_model(input_shape):
    """
    Build an LSTM model for video classification.
    Args:
        input_shape (tuple): Shape of the input data.
    Returns:
        keras.Model: Compiled LSTM model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# Main Script
processed_real_dir = "/home/ubuntu/DL_Project/processed_videos/real"
processed_fake_dir = "/home/ubuntu/DL_Project/processed_videos/fake"

real_videos = [os.path.join(processed_real_dir, f) for f in os.listdir(processed_real_dir) if f.endswith('.mp4')]
fake_videos = [os.path.join(processed_fake_dir, f) for f in os.listdir(processed_fake_dir) if f.endswith('.mp4')]

real_labels = [0] * len(real_videos)
fake_labels = [1] * len(fake_videos)

all_videos = real_videos + fake_videos
all_labels = real_labels + fake_labels

(train_videos, temp_videos, train_labels, temp_labels) = train_test_split(
    all_videos, all_labels, test_size=0.3, stratify=all_labels, random_state=42
)
(val_videos, test_videos, val_labels, test_labels) = train_test_split(
    temp_videos, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# Process Videos
print("Processing training videos...")
X_train, y_train = process_videos(train_videos, train_labels)
print("Processing validation videos...")
X_val, y_val = process_videos(val_videos, val_labels)
print("Processing test videos...")
X_test, y_test = process_videos(test_videos, test_labels)

# Pad Features
print("Padding features...")
X_train = pad_features(X_train)
X_val = pad_features(X_val, max_frames=X_train.shape[1])
X_test = pad_features(X_test, max_frames=X_train.shape[1])

# Build Model
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_lstm_model(input_shape)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, verbose=1)
]

# Train Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Classification Report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
f1_macro = f1_score(y_test, y_pred, average="macro")
print(f"F1 Macro Score: {f1_macro:.4f}")

# Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.show()

import tensorflow as tf
import pathlib
import cv2
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score, classification_report, recall_score, precision_score, confusion_matrix

# Constants
HEIGHT, WIDTH = 112, 112  # Resize dimensions for InceptionV3
N_FRAMES = 25            # Number of frames per video
FEATURE_DIM = 2048       # Feature dimension after InceptionV3
BATCH_SIZE = 25

# InceptionV3 Feature Extractor
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = models.Model(inputs=base_model.input, outputs=base_model.output)

# Preprocess Frames for InceptionV3
def preprocess_frame(frame):
    frame = cv2.resize(frame, (HEIGHT, WIDTH))
    frame = tf.keras.applications.inception_v3.preprocess_input(frame)  # Normalize for InceptionV3
    return frame

# Frame Generator with Feature Extraction
class FeatureGenerator:
    def __init__(self, video_paths, n_frames, training=False):
        """Generates feature vectors and labels for each video."""
        self.video_paths = video_paths
        self.n_frames = n_frames
        self.training = training

    def __call__(self):
        if self.training:
            np.random.shuffle(self.video_paths)

        for path, label in self.video_paths:
            features = self.extract_features_from_video(path, self.n_frames)
            yield features, label

    def extract_features_from_video(self, video_path, n_frames):
        """Extract features from video frames."""
        src = cv2.VideoCapture(str(video_path))
        video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = min(video_length, n_frames)
        result = []

        src.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start at the beginning
        for _ in range(max_frames):
            ret, frame = src.read()
            if ret:
                preprocessed_frame = preprocess_frame(frame)  # Resize and preprocess
                feature = feature_extractor.predict(np.expand_dims(preprocessed_frame, axis=0), verbose=0)
                result.append(feature.squeeze())  # Append extracted feature
            else:
                result.append(np.zeros((FEATURE_DIM,)))  # Pad with zeros if no frame

        while len(result) < n_frames:
            result.append(np.zeros((FEATURE_DIM,)))  # Pad to n_frames if needed

        src.release()
        return np.array(result)  # Shape: (n_frames, FEATURE_DIM)

# Paths for Real and Fake Videos
real_path = pathlib.Path("/home/ubuntu/Group6_DL_FTP/data/Celeb-real")
fake_path = pathlib.Path("/home/ubuntu/Group6_DL_FTP/data/Celeb-synthesis")


# Collect Video Paths and Labels
real_videos = list(real_path.glob('*.mp4'))
fake_videos = list(fake_path.glob('*.mp4'))
real_labels = [0] * len(real_videos)
fake_labels = [1] * len(fake_videos)

# Combine Data
all_videos = real_videos + fake_videos
all_labels = real_labels + fake_labels
video_label_pairs = list(zip(all_videos, all_labels))

# Train, Validation, and Test Splits
X_train_val, X_test, y_train_val, y_test = train_test_split(
    video_label_pairs, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
)

# Feature Generators
train_gen = FeatureGenerator(X_train, N_FRAMES, training=True)
val_gen = FeatureGenerator(X_val, N_FRAMES, training=False)
test_gen = FeatureGenerator(X_test, N_FRAMES, training=False)

# TF Datasets
train_ds = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        tf.TensorSpec(shape=(N_FRAMES, FEATURE_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    val_gen,
    output_signature=(
        tf.TensorSpec(shape=(N_FRAMES, FEATURE_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_generator(
    test_gen,
    output_signature=(
        tf.TensorSpec(shape=(N_FRAMES, FEATURE_DIM), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_lstm_model(n_frames, feature_dim, n_classes):
    input_shape = (n_frames, feature_dim)
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.LSTM(32),
        layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='sigmoid')  # Binary classification
    ])
    return model


# Focal Loss
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = -alpha_t * K.pow(1. - p_t, gamma) * K.log(p_t)
        return tf.reduce_mean(focal_loss)
    return focal_loss_fixed


# Define Model
n_classes = 1
model = build_lstm_model(N_FRAMES, FEATURE_DIM, n_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=focal_loss(gamma=2., alpha=0.25),
              metrics=['accuracy'])

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("cnn_lstm_best_model.keras", monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, verbose=1)
]

# Train Model
history = model.fit(
    train_ds.map(lambda x, y: (x, y)),  # Ignore video names for training
    validation_data=val_ds.map(lambda x, y: (x, y)),  # Ignore video names for validation
    epochs=10,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# Evaluate Model
test_loss, test_acc = model.evaluate(test_ds.map(lambda x, y, _: (x, y)))  # Ignore video names for evaluation
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Predictions and Misclassifications
y_true = []
y_pred = []
video_names = []

for frames, labels, names in test_ds:
    predictions = (model.predict(frames) > 0.5).astype(int).flatten()
    y_true.extend(labels.numpy())
    y_pred.extend(predictions)
    video_names.extend(names.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# F1 Score and Classification Report
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score: {f1:.4f}")
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
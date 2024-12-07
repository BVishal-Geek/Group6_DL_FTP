import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight, shuffle
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import tensorflow as tf
import tensorflow.keras.backend as K

# Constants
FRAME_SIZE = (299, 299)  # InceptionV3 expects 299x299 images
NUM_FEATURES = 2048  # Features extracted from InceptionV3's pooling layer
MAX_SEQ_LENGTH = 16

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


def extract_frame_features_with_augmentation(video_path, label, augment=False, max_frames=MAX_SEQ_LENGTH):
    """Extract features for a fixed number of frames from a video, with face detection and optional augmentation."""
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
                continue  # Skip frames that cause issues

    cap.release()

    # Ensure consistent frame count
    if len(frames) < max_frames:
        print(f"Skipping video {video_path} due to insufficient frames ({len(frames)}/{max_frames}).")
        return None  # Skip videos with fewer than max_frames

    return feature_extractor.predict(np.array(frames), verbose=0)




def process_videos_with_augmentation(video_paths, labels, augment_for_label=None, max_frames=MAX_SEQ_LENGTH):
    """Process video paths, extract features, and augment specific class."""
    X, y = [], []
    for video_path, label in zip(video_paths, labels):
        augment = label == augment_for_label  # Augment only for the specified class
        features = extract_frame_features_with_augmentation(video_path, label, augment=augment, max_frames=max_frames)
        if features is not None:  # Keep only valid feature arrays
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)


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

    overall_distribution = Counter(all_labels)
    print(f"Overall class distribution: {overall_distribution}")

    # Stratify the first split
    train_videos, temp_videos, train_labels, temp_labels = train_test_split(
        all_videos, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )

    # Stratify the second split (validation vs. test)
    val_videos, test_videos, val_labels, test_labels = train_test_split(
        temp_videos, temp_labels, test_size=val_size, random_state=random_state, stratify=temp_labels
    )

    return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)


def augment_frames(frames):
    """Apply data augmentation to frames using ImageDataGenerator."""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    augmented = [datagen.random_transform(frame) for frame in frames]
    return np.array(augmented)

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

def build_gru_model(input_shape=(MAX_SEQ_LENGTH, NUM_FEATURES)):
    """Build the GRU-based model."""
    frame_features_input = tf.keras.Input(shape=input_shape, name="frame_features")
    mask_input = tf.keras.Input(shape=(MAX_SEQ_LENGTH,), dtype="bool", name="mask")
    x = layers.GRU(64, return_sequences=True)(frame_features_input, mask=mask_input)
    x = layers.BatchNormalization()(x)
    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GRU(16)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=[frame_features_input, mask_input], outputs=output)
    model.compile(
        loss=focal_loss(gamma=2., alpha=0.25),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"]
    )
    return model


# Main Script
real_paths = ["/home/ubuntu/DL_Project/archive/all_real/"]
fake_paths = ["/home/ubuntu/DL_Project/archive/Celeb-synthesis/"]

# Step 1: Split videos into train, validation, and test sets
(train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels) = split_videos(
    real_paths, fake_paths
)

train_video_ids = {os.path.basename(video) for video in train_videos}
val_video_ids = {os.path.basename(video) for video in val_videos}
overlap = train_video_ids.intersection(val_video_ids)
assert len(overlap) == 0, f"Data leakage detected! Overlap: {overlap}"
# Check class distribution in validation set

val_distribution = Counter(val_labels)
print(f"Validation class distribution: {val_distribution}")
#%%
# Step 2: Process videos and extract features
# Process training videos (augment real videos for oversampling)
print("Processing training videos with augmentation for real videos...")
X_train, y_train = process_videos_with_augmentation(train_videos, train_labels, augment_for_label=0)

# Process validation and test videos (no augmentation)
print("Processing validation videos...")
X_val, y_val = process_videos_with_augmentation(val_videos, val_labels)

print("Processing test videos...")
X_test, y_test = process_videos_with_augmentation(test_videos, test_labels, augment_for_label=0)
#%%

# Step 3: Create masks
train_mask = np.ones((len(X_train), MAX_SEQ_LENGTH), dtype=bool)
val_mask = np.ones((len(X_val), MAX_SEQ_LENGTH), dtype=bool)
test_mask = np.ones((len(X_test), MAX_SEQ_LENGTH), dtype=bool)

# Step 4: Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")

# Step 5: Build and train the GRU model
model = build_gru_model()
print("Training the model...")
history = model.fit(
    [X_train, train_mask], y_train,
    validation_data=([X_val, val_mask], y_val),
    epochs=10,
    batch_size=16,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True)
    ]
)

# Step 6: Evaluate the model
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate([X_test, test_mask], y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 7: Classification report and F1 score
predictions = model.predict([X_test, test_mask])
y_pred = (predictions > 0.5).astype(int)
print(classification_report(y_test, y_pred))
f1_macro = f1_score(y_test, y_pred, average="macro")
print(f"F1 Macro Score: {f1_macro:.4f}")




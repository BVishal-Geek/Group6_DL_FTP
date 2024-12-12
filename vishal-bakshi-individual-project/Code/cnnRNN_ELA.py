import os
import numpy as np
import cv2
from six import BytesIO
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
from PIL import Image, ImageChops

# Constants
FRAME_SIZE = (299, 299)  # InceptionV3 expects 299x299 images
NUM_FEATURES = 2048  # Features extracted from InceptionV3's pooling layer
MAX_SEQ_LENGTH = 5

# Load Pretrained Feature Extractor (InceptionV3)
feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

def generate_ela_images(frame, quality):
    """Converts the image/frame into 'Error Loss Analysis' format to understand the difference in inconsistenices"""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)

    ela_image = ImageChops.difference(pil_image, compressed_image)
    ela_image = ela_image.point(lambda x: x*10)
    ela_frame = cv2.cvtColor(np.array(ela_image), cv2.COLOR_RGB2BGR)

    return ela_frame


def extract_frame_features_with_augmentation(video_path, label, augment=False, max_frames=MAX_SEQ_LENGTH, quality=95):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return None, None

    frames = []
    ela_frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"No valid frames in video {video_path}")
        return None, None

    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Frame read failed in video {video_path} at frame {i}")
            continue
        if i in frame_indices:
            try:
                resized_frame = cv2.resize(frame, FRAME_SIZE)
                frame = preprocess_input(resized_frame)
                frames.append(frame)

                ela_frame = generate_ela_images(resized_frame, quality=quality)
                ela_frame = cv2.resize(ela_frame, FRAME_SIZE)
                ela_frame = preprocess_input(ela_frame)
                ela_frames.append(ela_frame)
            except Exception as e:
                print(f"Error processing frame in {video_path} at frame {i}: {e}")
                continue

    cap.release()

    if len(frames) < max_frames or not frames or not ela_frames:
        print(f"Skipping video {video_path} due to insufficient or empty frames.")
        return None, None

    try:
        regular_features = feature_extractor.predict(np.array(frames), verbose=0)
        ela_features = feature_extractor.predict(np.array(ela_frames), verbose=0)
    except Exception as e:
        print(f"Error during feature extraction for {video_path}: {e}")
        return None, None

    return regular_features, ela_features




def process_videos_with_augmentation(video_paths, labels, augment_for_label=None, max_frames=MAX_SEQ_LENGTH):
    """Process video paths, extract features, and augment specific class."""
    X, X_ela, y = [], [], []
    for video_path, label in zip(video_paths, labels):
        augment = label == augment_for_label  # Augment only for the specified class
        features, ela_features = extract_frame_features_with_augmentation(video_path, label, augment=augment, max_frames=max_frames)
        if features is not None and ela_features is not None:  # Keep only valid feature arrays
            X.append(features)
            X_ela.append(ela_features)
            y.append(label)
    return np.array(X), np.array(X_ela), np.array(y)



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
real_paths = ["/home/ubuntu/Group6_DL_FTP/data-processed/Celeb-real-processed/"]
fake_paths = ["/home/ubuntu/Group6_DL_FTP/data-processed/Celeb-fake-processed/"]

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
X_train, X_train_ela, y_train = process_videos_with_augmentation(train_videos, train_labels, augment_for_label=0)
# Save the oversampled training data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_train_ela.npy", X_train_ela)
# Load the data in subsequent runs to save time
# X_train = np.load("X_train.npy")
# y_train = np.load("y_train.npy")
# X_train_ela = np.load("X_train_ela.npy")
# Process validation and test videos (no augmentation)

print("Processing validation videos...")
X_val, X_val_ela, y_val = process_videos_with_augmentation(val_videos, val_labels)


print("Processing test videos...")
X_test, X_test_ela, y_test = process_videos_with_augmentation(test_videos, test_labels)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
np.save("X_test_ela.npy", X_test_ela)

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
    [X_train, X_train_ela, train_mask],  # Regular features, ELA features, and masks
    y_train,
    validation_data=([X_val, X_val_ela, val_mask], y_val),  # Validation data
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
test_loss, test_accuracy = model.evaluate([X_test, X_test_ela,test_mask], y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 7: Classification report and F1 score
predictions = model.predict([X_test, X_test_ela ,test_mask])
y_pred = (predictions > 0.5).astype(int)
print(classification_report(y_test, y_pred))
f1_macro = f1_score(y_test, y_pred, average="macro")
print(f"F1 Macro Score: {f1_macro:.4f}")
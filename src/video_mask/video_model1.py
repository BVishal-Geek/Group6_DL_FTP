import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, Input, Model
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

# Base directory for video data
video_base_dir = '/home/ubuntu/Final_Project/data'  # Update this path


# Function to extract frames from video at 50 fps
def extract_frames(video_path, fps=2):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Ensure original_fps is valid and greater than zero
    if original_fps > 0:
        frame_interval = max(1, int(original_fps / fps))
    else:
        print(f"Warning: Unable to retrieve FPS for {video_path}. Skipping video.")
        return []

    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames


# Function to calculate auxiliary features
def calculate_average_pixel_value(image):
    return np.mean(image)


# Load video data and extract frames and features
def build_vgg_model_with_aux(input_shape):
    image_input = Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(image_input)
    #x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    #x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    #x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and check shape
    flat_x = layers.Flatten()(x)

    aux_input = Input(shape=(1,))

    combined_input = layers.concatenate([flat_x, aux_input])

    # Adjust dense layer input size
    combined_output = layers.Dense(57603, activation='relu')(combined_input)  # Adjust this size if needed
    combined_output = layers.Dense(4096, activation='relu')(combined_output)

    final_output = layers.Dense(1, activation='sigmoid')(combined_output)

    model = Model(inputs=[image_input, aux_input], outputs=[final_output])

    return model


# Ensure frames are resized correctly
# Load video data and extract frames and features
def load_video_data(folders):
    all_frames = []
    all_labels = []
    all_aux_features = []

    for folder in folders:
        folder_path = os.path.join(video_base_dir, folder)
        label = 1 if 'real' in folder else 0

        for video_file in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_file)
            frames = extract_frames(video_path)

            for frame in frames:
                frame_resized = cv2.resize(frame, (64, 64))
                all_frames.append(frame_resized)
                all_labels.append(label)
                aux_feature = calculate_average_pixel_value(frame_resized)
                all_aux_features.append(aux_feature)

    # Ensure all arrays have the same length
    min_length = min(len(all_frames), len(all_labels), len(all_aux_features))
    all_frames = np.array(all_frames[:min_length])
    all_labels = np.array(all_labels[:min_length])
    all_aux_features = np.array(all_aux_features[:min_length]).reshape(-1, 1)

    return all_frames, all_labels, all_aux_features

# Load data from folders
folders = ['Celeb-real', 'Celeb-synthesis']
frames, labels, aux_features = load_video_data(folders)

# Check lengths before splitting
print(f"Frames: {len(frames)}, Labels: {len(labels)}, Aux Features: {len(aux_features)}")

# Split data into train, test, and validation sets
X_train, X_temp, y_train, y_temp, aux_train, aux_temp = train_test_split(
    frames,
    labels,
    aux_features,
    test_size=0.4,
    random_state=42
)

X_val, X_test, y_val, y_test, aux_val, aux_test = train_test_split(
    X_temp,
    y_temp,
    aux_temp,
    test_size=0.5,
    random_state=42
)

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weights = alpha * y_true + (1 - alpha) * (1 - y_true)
        focal_loss = weights * tf.pow((1 - y_pred), gamma) * cross_entropy
        return tf.reduce_mean(focal_loss)

    return focal_loss_fixed

# Build and compile the model
model = build_vgg_model_with_aux((64, 64, 3))
model.compile(optimizer=optimizers.Adam(), loss=focal_loss(), metrics=['accuracy'])

# Training configuration
batch_size = 1
epochs = 20

# Train the model using combined datasets and auxiliary features
history = model.fit(
    [X_train,
     aux_train],
    y_train,
    validation_data=([X_val,
                      aux_val], y_val),
    epochs=epochs,
    batch_size=batch_size,
)

# Evaluate the model on test data using combined datasets and auxiliary features
predictions = model.predict([X_test,
                             aux_test])
predicted_classes = (predictions > 0.35).astype(int)[:, 0]

# Calculate accuracy and F1 score on test data
accuracy = accuracy_score(y_test, predicted_classes)
f1 = f1_score(y_test, predicted_classes)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
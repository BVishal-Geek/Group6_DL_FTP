import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Input, Model
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.metrics import accuracy_score, f1_score

# Set the global policy to mixed precision
set_global_policy('mixed_float16')

# Base directory for video data
video_base_dir = '/home/ubuntu/Final_Project/data'  # Update this path

# Function to extract frames from video at specified fps
def extract_frames(video_path, fps=10):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)

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

# Build VGG-like model with auxiliary input
def build_vgg_model_with_aux(input_shape):
    image_input = Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(image_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    flat_x = layers.Flatten()(x)

    aux_input = Input(shape=(1,))
    combined_input = layers.concatenate([flat_x, aux_input])

    combined_output = layers.Dense(4096, activation='relu')(combined_input)
    final_output = layers.Dense(1, activation='sigmoid', dtype='float32')(combined_output)

    model = Model(inputs=[image_input, aux_input], outputs=[final_output])

    return model

# Load video data and extract frames and features as a tf.data.Dataset
def load_video_data_as_dataset(folders):
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
                frame_resized = cv2.resize(frame, (264, 264))
                all_frames.append(frame_resized)
                all_labels.append(label)
                aux_feature = calculate_average_pixel_value(frame_resized)
                all_aux_features.append(aux_feature)

    min_length = min(len(all_frames), len(all_labels), len(all_aux_features))

    all_frames_np = np.array(all_frames[:min_length])
    all_labels_np = np.array(all_labels[:min_length])
    all_aux_features_np = np.array(all_aux_features[:min_length]).reshape(-1, 1)

    dataset = tf.data.Dataset.from_tensor_slices(((all_frames_np.astype('float16'),
                                                   all_aux_features_np.astype('float16')),
                                                  all_labels_np.astype('float16')))

    return dataset

# Load data from folders and create dataset
folders = ['Celeb-real', 'Celeb-synthesis']
dataset = load_video_data_as_dataset(folders)

# Shuffle with a fixed seed for consistent results
dataset_size = len(list(dataset))
dataset = dataset.shuffle(buffer_size=dataset_size, seed=42)

# Split data into train, validation, and test datasets
train_size = int(0.6 * dataset_size)
val_size = int(0.2 * dataset_size)

train_dataset = dataset.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = dataset.skip(train_size).take(val_size).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = dataset.skip(train_size + val_size).batch(32).prefetch(tf.data.AUTOTUNE)

# Verify dataset sizes after splitting
print(f"Train dataset size: {len(list(train_dataset))}")
print(f"Validation dataset size: {len(list(val_dataset))}")
print(f"Test dataset size: {len(list(test_dataset))}")

# Inspect some samples from each dataset to ensure correct splitting
for element in train_dataset.take(1):
    print("Train sample:", element)

for element in val_dataset.take(1):
    print("Validation sample:", element)

for element in test_dataset.take(1):
    print("Test sample:", element)

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred_clipped = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy_loss = -y_true * tf.math.log(y_pred_clipped)
        weights = alpha * y_true + (1 - alpha) * (1 - y_true)
        focal_loss_value = weights * tf.pow((1 - y_pred_clipped), gamma) * cross_entropy_loss
        return tf.reduce_mean(focal_loss_value)

    return focal_loss_fixed

# Build and compile the model
model = build_vgg_model_with_aux((264, 264, 3))
model.compile(optimizer=optimizers.Adam(), loss=focal_loss(), metrics=['accuracy'])

# Train the model using the dataset
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
)

# Evaluate the model on test data using combined datasets and auxiliary features
predictions_float32_test_dataset = []
y_test = []
for batch in test_dataset:
    predictions_float32_test_dataset.append(model.predict(batch[0]))
    y_test.append(batch[1].numpy())

predicted_classes = []
for prediction in predictions_float32_test_dataset:
    predicted_classes.extend((prediction > 0.25).astype(int)[:, 0])

y_test = np.concatenate(y_test).astype(int)

# Calculate accuracy and F1 score on test data
accuracy = float(accuracy_score(y_test, predicted_classes))
f1 = float(f1_score(y_test, predicted_classes))

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
import os
import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check GPU availability
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#%%

# Helper Functions
def extract_frames(video_path, label, output_dir, max_frames=10):
    """Extract frames from a video and save them to disk."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_paths = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))

        # Save frame to disk
        frame_path = os.path.join(output_dir, f"{base_name}_frame{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append((frame_path, label))
        frame_count += 1

    cap.release()
    return frame_paths


def load_data_to_disk(paths, label, output_dir, batch_size=100):
    """Load video data in batches and save frames to disk."""
    os.makedirs(output_dir, exist_ok=True)
    data = []
    batch_count = 0

    for path in paths:
        print(f"Processing path: {path}")
        videos = os.listdir(path)
        for i in range(0, len(videos), batch_size):
            batch_videos = videos[i:i + batch_size]
            for video in batch_videos:
                video_path = os.path.join(path, video)
                data.extend(extract_frames(video_path, label, output_dir))
            print(f"Processed batch {batch_count}")
            batch_count += 1

    return data


def get_video_ids(data):
    """Extract unique video IDs from frame paths."""
    return [os.path.basename(item[0]).split('_')[0] for item in data]


def split_data_at_video_level(data):
    """Split data at the video level to avoid leakage."""
    video_ids = list(set(get_video_ids(data)))
    train_ids, test_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

    train_data = [item for item in data if os.path.basename(item[0]).split('_')[0] in train_ids]
    test_data = [item for item in data if os.path.basename(item[0]).split('_')[0] in test_ids]

    return train_data, test_data


def data_generator(data, batch_size, class_weights):
    """Generator to yield batches of data with sample weights."""
    while True:
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]

            # Load images and labels
            batch_images = [cv2.imread(item[0]) for item in batch_data]
            batch_images = [cv2.resize(img, (224, 224)) for img in batch_images]
            batch_images = np.array(batch_images) / 255.0  # Normalize
            batch_labels = np.array([item[1] for item in batch_data])

            # Compute sample weights based on class weights
            batch_weights = np.array([class_weights[label] for label in batch_labels])

            yield batch_images, batch_labels, batch_weights


def build_model():
    """Build the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model_with_generator(model, train_data, test_data, batch_size, class_weights, epochs=10):
    """Train the model using a data generator with sample weights."""
    train_gen = data_generator(train_data, batch_size, class_weights)
    test_gen = data_generator(test_data, batch_size, class_weights)

    steps_per_epoch = len(train_data) // batch_size
    validation_steps = len(test_data) // batch_size

    start_time = time.time()
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_gen,
        validation_steps=validation_steps
    )
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    return history

#%%
from sklearn.metrics import classification_report, f1_score
def evaluate_model(model, test_data, batch_size, class_weights):
    """Evaluate the model and calculate F1-macro score."""
    test_gen = data_generator(test_data, batch_size, class_weights)
    steps = len(test_data) // batch_size

    # Get predictions
    start_time = time.time()
    predictions = model.predict(test_gen, steps=steps)
    predictions = np.round(predictions).flatten()  # Convert probabilities to binary (0 or 1)
    end_time = time.time()

    # Extract true labels
    _, y_test = zip(*test_data)
    y_test = np.array(y_test[:len(predictions)])  # Ensure labels match the number of predictions

    # Calculate F1-macro score
    f1_macro = f1_score(y_test, predictions, average='macro')

    # Print results
    print(f"Test Accuracy: {np.mean(predictions == y_test) * 100:.2f}%")
    print(f"Test F1 Macro Score: {f1_macro * 100:.2f}%")
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")

    return f1_macro


#%%
# Main Script
# Paths to your datasets
real_paths = ["/home/ubuntu/DL_Project/archive/Celeb-real/",
              "/home/ubuntu/DL_Project/archive/YouTube-real/"]
synthetic_paths = ["/home/ubuntu/DL_Project/archive/Celeb-synthesis/"]
#%%
# Load data
real_data = load_data_to_disk(real_paths, 0, "/tmp/real_frames/")
synthetic_data = load_data_to_disk(synthetic_paths, 1, "/tmp/synthetic_frames/")
all_data = real_data + synthetic_data

# Split data at video level
train_data, test_data = split_data_at_video_level(all_data)
#%%
# Compute class weights
_, y_train = zip(*train_data)
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
#%%
# Build and train the model
my_model = build_model()
batch_size = 32
train_history = train_model_with_generator(my_model, train_data, test_data, batch_size, class_weights, epochs=10)
#%%
# Evaluate the model
f1_macro_score = evaluate_model(my_model, test_data, batch_size, class_weights)


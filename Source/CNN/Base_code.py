#%%
import os
import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
 #%%
def extract_frames(video_path, label, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        yield np.array(frame), label
        frame_count += 1
    cap.release()

#%%
def load_data_in_batches(paths, label, batch_size=100):
    start_time = time.time()
    data = []
    batch_count = 0

    for path in paths:
        print(f"Processing path: {path}")
        videos = os.listdir(path)
        for i in range(0, len(videos), batch_size):
            batch_videos = videos[i:i + batch_size]
            for video in batch_videos:
                video_path = os.path.join(path, video)
                data.extend(extract_frames(video_path, label))

            # Process the batch and clear memory
            yield data
            data = []
            batch_count += 1
            print(f"Processed batch {batch_count}")

    end_time = time.time()
    print(f"Loaded data in batches from {paths}. Time taken: {end_time - start_time:.2f} seconds")


#%%
# Paths to your datasets

real_paths = ["/home/ubuntu/DL_Project/archive/Celeb-real/",
              "/home/ubuntu/DL_Project/archive/YouTube-real/"]
synthetic_paths = ["/home/ubuntu/DL_Project/archive/Celeb-synthesis/"]

# Load and label data
real_data = []
for batch in load_data_in_batches(real_paths, 0):  # Label 0 for real
    real_data.extend(batch)

synthetic_data = []
for batch in load_data_in_batches(synthetic_paths, 1):  # Label 1 for synthetic
    synthetic_data.extend(batch)

#print(real_data)
#print(synthetic_data)
#%%
# Combine and split data
all_data = real_data + synthetic_data
half_size = len(all_data)//4
subset_data = all_data[:half_size]
X, y = zip(*subset_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Data Augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(np.array(X_train), np.array(y_train), batch_size=32)
#%%
def build_model():
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

my_model = build_model()
# Compute class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Train the model
def train_model(model, train_generator, X_test, y_test, epochs=2):
    start_time = time.time()
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // 32,
        epochs=epochs,
        validation_data=(np.array(X_test) / 255.0, np.array(y_test)),
        class_weight=class_weights
    )
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    return history

train_history = train_model(my_model, train_generator, X_test, y_test, epochs=10)
#%%
# Evaluate the Model
def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    loss, accuracy = model.evaluate(np.array(X_test) / 255.0, np.array(y_test))
    end_time = time.time()
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    #print(f"Test F1 Score: {f1_macro*100:.2f}%")
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")

evaluate_model(my_model, X_test, y_test)


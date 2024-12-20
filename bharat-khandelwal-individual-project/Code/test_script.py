import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model

# Constants
FRAME_SIZE = (299, 299)  # InceptionV3 expects 299x299 images
NUM_FEATURES = 2048  # Features extracted from InceptionV3's pooling layer
MAX_SEQ_LENGTH = 16

# Load Pretrained Feature Extractor (InceptionV3)
feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

# Define helper functions
def crop_center_square(frame):
    """Crop the center square of the frame."""
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def test_individual_video(video_path, model, feature_extractor, max_frames=MAX_SEQ_LENGTH):
    """Test the trained model on a new individual video."""
    # Extract features from the video
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            # Crop and resize the frame
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, FRAME_SIZE)
            frame = preprocess_input(frame)
            frames.append(frame)
    cap.release()

    frames = np.array(frames)
    if len(frames) < max_frames:
        print("Error: Video has fewer frames than the required sequence length.")
        return None

    # Extract features using the feature extractor
    features = feature_extractor.predict(frames, verbose=0)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    mask = np.ones((1, MAX_SEQ_LENGTH), dtype=bool)  # Create a mask

    # Make predictions
    prediction = model.predict([features, mask])
    label = int(prediction > 0.5)  # Threshold at 0.5
    label_name = "Fake" if label == 1 else "Real"

    print(f"Prediction: {label_name} ({prediction[0][0]:.4f})")
    return label_name, prediction[0][0]
#%%
# Main testing functionality
if __name__ == "__main__":
    # Path to the new video
    new_video_path = "/Users/bharatkhandelwal/Downloads/test.mp4"

    # Load the trained model
    model = load_model("best_model.keras", compile=False)

    # Test the new video
    label_name, confidence = test_individual_video(new_video_path, model, feature_extractor)
    print(f"The video is classified as {label_name} with confidence {confidence:.4f}.")

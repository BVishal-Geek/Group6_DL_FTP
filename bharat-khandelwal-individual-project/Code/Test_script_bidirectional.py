import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# Constants
FRAME_SIZE = (299, 299)  # InceptionV3 expects 299x299 images
NUM_FEATURES = 2048  # Features extracted from InceptionV3's pooling layer
MAX_SEQ_LENGTH = 16

# Load Pretrained Feature Extractor (InceptionV3)
feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define helper functions
def crop_center_square(frame):
    """Crop the center square of the frame."""
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def crop_face(frame):
    """Detect and crop the face from the frame."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = frame[y:y + h, x:x + w]
        return cv2.resize(face, FRAME_SIZE)
    else:
        return crop_center_square(frame)

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
            try:
                frame = crop_face(frame)
                frame = cv2.resize(frame, FRAME_SIZE)
                frame = preprocess_input(frame)
                frames.append(frame)
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
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

# Define the focal loss function again to ensure it's recognized
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

def custom_lambda_layer(x):
    return tf.reduce_sum(x, axis=1)

#%%
# Main testing functionality
if __name__ == "__main__":
    # Path to the new video
    new_video_path = input("Enter the path to the video: ").strip()

    # Verify that the video exists
    if not os.path.exists(new_video_path):
        print(f"Error: Video file '{new_video_path}' does not exist.")
    else:
        # Load the trained model
        model = load_model(
            "best_model_bidirectional.keras",
            custom_objects={
                'custom_lambda_layer': custom_lambda_layer
            },
            compile=False,
            safe_mode=False)

        # Test the new video
        label_name, confidence = test_individual_video(new_video_path, model, feature_extractor)
        if label_name is not None:
            print(f"The video is classified as {label_name} with confidence {confidence:.4f}.")

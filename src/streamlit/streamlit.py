import os
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.models import load_model

# Constants for Models
FRAME_SIZE_INCEPTION = (299, 299)  # InceptionV3 expects 299x299 images
FRAME_SIZE_VGG = (224, 224)  # VGG16 expects 224x224 images
MAX_SEQ_LENGTH = 16  # Number of frames to process

# Force CPU mode if GPU initialization fails (temporary fix)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load Pretrained Feature Extractor for InceptionV3
try:
    feature_extractor_inception = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
except Exception as e:
    st.error(f"Error initializing InceptionV3: {e}")

st.set_page_config(page_title="Forgery Detection - Group 6", layout="wide")

# Helper Functions
def crop_center_square(frame):
    """Crop the center square of the frame."""
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def preprocess_video_for_inception(video_path, max_frames=MAX_SEQ_LENGTH):
    """Preprocess a video for InceptionV3."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < max_frames:
        st.error(f"Video has only {total_frames} frames but requires at least {max_frames}.")
        return None

    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            # Crop and resize the frame
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, FRAME_SIZE_INCEPTION)
            frame = inception_preprocess_input(frame)
            frames.append(frame)

    cap.release()

    if len(frames) < max_frames:
        st.error(f"Insufficient frames after preprocessing. Collected {len(frames)} frames.")
        return None

    frames = np.array(frames)

    # Extract features using the feature extractor
    try:
        features = feature_extractor_inception.predict(frames, verbose=0)
    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None

    features = np.expand_dims(features, axis=0)  # Add batch dimension
    mask = np.ones((1, MAX_SEQ_LENGTH), dtype=bool)  # Create a mask

    return features, mask


def preprocess_video_for_vgg(video_path, max_frames=MAX_SEQ_LENGTH):
    """Preprocess a video for VGG16."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < max_frames:
        st.error(f"Video has only {total_frames} frames but requires at least {max_frames}.")
        return None

    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            # Crop and resize the frame
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, FRAME_SIZE_VGG)
            frame = vgg_preprocess_input(frame)
            frames.append(frame)

    cap.release()

    if len(frames) < max_frames:
        st.error(f"Insufficient frames after preprocessing. Collected {len(frames)} frames.")
        return None

    frames = np.array(frames)

    return frames


def test_video_with_vgg(video_path, model):
    """Test a video using the trained VGG16-based model."""
    frames = preprocess_video_for_vgg(video_path)

    if frames is None:
        return None

    predictions = []

    for frame in frames:  # Iterate over individual frames
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension (1, 224, 224, 3)
        prediction = model.predict(frame, verbose=0)  # Predict for single frame
        predictions.append(prediction[0][0])  # Extract probability for positive class

    avg_prediction = np.mean(predictions)  # Aggregate predictions (average probability)

    label = int(avg_prediction > 0.5)  # Threshold at 0.5
    label_name = "Fake" if label == 1 else "Real"

    return label_name, avg_prediction


def test_video_with_inception(video_path, model):
    """Test a video using the trained InceptionV3-based model."""
    data = preprocess_video_for_inception(video_path)

    if data is None:
        return None

    features, mask = data

    try:
        prediction = model.predict([features, mask])
        label = int(prediction > 0.5)  # Threshold at 0.5
        label_name = "Fake" if label == 1 else "Real"
        return label_name, prediction[0][0]
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None


# Implementation Page Functionality
def implementation_page():
    st.title("Implementation - Forgery Detection")

    # Dropdown menu to select the model
    model_options = ["Select a Model", "InceptionV3", "VGG16"]
    selected_model = st.selectbox("Choose a Model:", model_options)

    if selected_model != "Select a Model":
        st.write(f"You selected: {selected_model}")

        # File uploader for video input
        uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_file is not None:
            temp_video_path = f"temp_{uploaded_file.name}"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.read())

            st.success("Video uploaded successfully!")

            if selected_model == "InceptionV3":
                st.info("Running InceptionV3 model...")
                inception_model_path = "best_model.keras"
                try:
                    model_inception = load_model(inception_model_path, compile=False)
                    result_inception = test_video_with_inception(temp_video_path, model_inception)

                    if result_inception is not None:
                        label_name_incep, confidence_incep = result_inception
                        st.success(
                            f"The video is classified as **{label_name_incep}** with confidence **{confidence_incep:.4f}**.")
                    else:
                        st.error("Failed to process the video.")
                except Exception as e:
                    st.error(f"Error loading InceptionV3 model: {e}")

            elif selected_model == "VGG16":
                st.info("Running VGG16 model...")
                vgg_model_path = "finetuned_vgg16.keras"
                try:
                    model_vgg16 = load_model(vgg_model_path)
                    result_vgg16 = test_video_with_vgg(temp_video_path, model_vgg16)

                    if result_vgg16 is not None:
                        label_name_vgg, confidence_vgg = result_vgg16
                        st.success(
                            f"The video is classified as **{label_name_vgg}** with confidence **{confidence_vgg:.4f}**.")
                    else:
                        st.error("Failed to process the video.")
                except Exception as e:
                    st.error(f"Error loading VGG16 model: {e}")


# Introduction Page Functionality
def introduction_page():
    st.title("Forgery Detection - Group 6")

    # Project Overview
    st.subheader("Project Overview")
    st.markdown("""
    - Every day, approximately **3.2 billion images** and **720,000 hours of video** are shared online. [Reference](https://www.qut.edu.au/insights/business/3.2-billion-images-and-720000-hours-of-video-are-shared-online-daily.-can-you-sort-real-from-fake)
    - The rise of deepfakes has led to a significant increase in **morphed videos and images**, causing misinformation and trust issues globally.
    - Misinformation through manipulated media can have **massive societal impacts**, including political, financial, and personal consequences.
    - Our project aims to address this issue by building robust models to detect forgery in multimedia content.
    """)

    # Dataset Information
    st.subheader("Dataset Information")
    st.markdown("""
    - We are using the **CelebDF dataset** for training and evaluation. [Reference](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.pdf)
    - The dataset includes videos categorized into:
      - Real videos (e.g., celebrity videos)
      - Fake videos (deepfake-generated content)
      - YouTube videos (real-world examples)
    """)

    # Learning Curve
    st.subheader("Learning Curve")
    st.markdown("""
    - Started with exploring datasets and research papers from repositories like [Awesome Deepfakes Detection](https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection?tab=readme-ov-file).
    - Worked on image datasets to understand how to load, segment, and preprocess data for real and fake classes.
    - Performed exploratory data analysis (EDA) on datasets to understand their structure before implementation.
      - For example, in video datasets, we ensured segmentation of train-test splits such that the test set contains unseen data.
      - Calculated mean pixel values for real and fake images, observing differences in red and blue channels.
      - Used these pixel means as auxiliary features in the fully connected layer of our models, which improved accuracy significantly.
    - Reviewed research papers like [CelebDF: A Large-Scale Challenging Dataset for DeepFake Forensics](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.pdf) to explore architectures used by other researchers.
      - Integrated insights into our implementation to build robust models.
    """)

    # Future Work
    st.subheader("Future Work")
    st.markdown("""
    - Generate face meshes for detected faces and analyze changes in node values on the mesh.
    - Implement masking techniques to highlight differences between real and altered images as auxiliary features.
    - Explore multi-modal approaches combining video and audio streams for forgery detection.
    """)


# About Us Page Functionality
def about_us_page():
    st.title("About Us - Group 6")

    team_members = [
        "Khush Shah",
        "Nena Beecham",
        "Vishal Bakshi",
        "Bharat Khandelwal"
    ]

    st.write("We are a team of researchers working on multimedia security and authenticity detection.")

    st.subheader("Team Members:")

    for member in team_members:
        st.write(f"- {member}")


# Sidebar Navigation Menu
menu = ["Introduction", "Implementation", "About Us"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Introduction":
    introduction_page()
elif choice == "Implementation":
    implementation_page()
elif choice == "About Us":
    about_us_page()
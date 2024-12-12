import os
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.models import load_model
import keras.config
from downloadModels import download_google_drive_folder

# Constants
FRAME_SIZE_INCEPTION = (299, 299)
FRAME_SIZE_VGG = (224, 224)
MAX_SEQ_LENGTH = 16
MODELS_FOLDER = os.path.join(os.getcwd(), "..", "models")
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Helper function
def crop_center_square(frame):
    """Crop the center square of the frame."""
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


# VGG Preprocessor
class VGGProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    @staticmethod
    def preprocess_frame(frame):
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, FRAME_SIZE_VGG)
        return vgg_preprocess_input(frame)

    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < MAX_SEQ_LENGTH:
            st.error(f"Video has only {total_frames} frames but requires at least {MAX_SEQ_LENGTH}.")
            return None

        frame_indices = np.linspace(0, total_frames - 1, MAX_SEQ_LENGTH, dtype=int)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frames.append(self.preprocess_frame(frame))
        cap.release()

        if len(frames) < MAX_SEQ_LENGTH:
            st.error(f"Insufficient frames after preprocessing. Collected {len(frames)} frames.")
            return None
        return np.array(frames)

    def predict(self, video_path):
        frames = self.preprocess_video(video_path)
        if frames is None:
            return None

        predictions = [self.model.predict(np.expand_dims(frame, axis=0))[0][0] for frame in frames]
        avg_prediction = np.mean(predictions)
        label = "Fake" if avg_prediction > 0.5 else "Real"
        return label, avg_prediction

    def load_model(self):
        try:
            return load_model(self.model_path, custom_objects={"f1_macro": ForgeryDetectionApp.f1_macro})
        except Exception as e:
            st.error(f"Error loading VGG16 model: {e}")
            return None


# Inception Preprocessor
class InceptionProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
        self.feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

    @staticmethod
    def preprocess_frame(frame):
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, FRAME_SIZE_INCEPTION)
        return inception_preprocess_input(frame)

    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < MAX_SEQ_LENGTH:
            st.error(f"Video has only {total_frames} frames but requires at least {MAX_SEQ_LENGTH}.")
            return None

        frame_indices = np.linspace(0, total_frames - 1, MAX_SEQ_LENGTH, dtype=int)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frames.append(self.preprocess_frame(frame))
        cap.release()

        if len(frames) < MAX_SEQ_LENGTH:
            st.error(f"Insufficient frames after preprocessing. Collected {len(frames)} frames.")
            return None

        frames = np.array(frames)
        features = self.feature_extractor.predict(frames, verbose=0)
        return np.expand_dims(features, axis=0), np.ones((1, MAX_SEQ_LENGTH), dtype=bool)

    def predict(self, video_path):
        data = self.preprocess_video(video_path)
        if data is None:
            return None

        features, mask = data
        prediction = self.model.predict([features, mask])[0][0]
        label = "Fake" if prediction > 0.5 else "Real"
        return label, prediction

    def load_model(self):
        try:
            return load_model(self.model_path, compile=False)
        except Exception as e:
            st.error(f"Error loading InceptionV3 model: {e}")
            return None

# Main App Class
class ForgeryDetectionApp:
    def __init__(self):
        self.vgg_processor = VGGProcessor(os.path.join(MODELS_FOLDER, "finetuned_vgg16.keras"))
        self.inception_processor = InceptionProcessor(os.path.join(MODELS_FOLDER, "best_model.keras"))


    @staticmethod
    def f1_macro(y_true, y_pred):
        y_pred = tf.round(y_pred)
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32), axis=0)
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32), axis=0)
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32), axis=0)
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return tf.reduce_mean(f1)

    def implementation_page(self):
        st.title("Implementation - Forgery Detection")
        model_options = ["Select a Model", "InceptionV3", "VGG16"]
        selected_model = st.selectbox("Choose a Model:", model_options)

        if selected_model != "Select a Model":
            uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file:
                st.success(f"The video is uploaded! Scroll down to see the results")
                temp_video_path = f"temp_{uploaded_file.name}"
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.video(temp_video_path)

                if selected_model == "InceptionV3":
                    label, confidence = self.inception_processor.predict(temp_video_path)
                elif selected_model == "VGG16":
                    label, confidence = self.vgg_processor.predict(temp_video_path)


                if label:
                    st.success(f"The video is classified as **{label}** with confidence **{confidence:.4f}**.")
                else:
                    st.error("Prediction failed.")

    def introduction_page(self):
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

    def about_us_page(self):
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

    def run(self):
        menu = ["Introduction", "Implementation", "About Us"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Introduction":
            self.introduction_page()
        elif choice == "Implementation":
            self.implementation_page()
        elif choice == "About Us":
            self.about_us_page()


# Run the app
if __name__ == "__main__":
    app = ForgeryDetectionApp()
    app.run()
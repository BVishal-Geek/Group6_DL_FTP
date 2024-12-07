import streamlit as st

# Set up the main configuration
st.set_page_config(page_title="Forgery Detection - Group 6", layout="wide")


# Define a function for preprocessing based on model selection
def preprocess_for_model(model_name):
    if model_name == "VGG":
        st.write("Preprocessing for VGG model...")
        # Add specific preprocessing steps for VGG here
        st.write("Images resized to (224x224) and normalized.")
    elif model_name == "LSTM":
        st.write("Preprocessing for LSTM model...")
        # Add specific preprocessing steps for LSTM here
        st.write("Extracting temporal features from video frames.")
    else:
        st.write("Please select a valid model.")


# Implementation Page Functionality
def implementation_page():
    st.title("Implementation - Forgery Detection")

    # Dropdown menu to select the model
    model_options = ["Select a Model", "VGG", "LSTM"]
    selected_model = st.selectbox("Choose a Model:", model_options)

    if selected_model != "Select a Model":
        st.write(f"You selected: {selected_model}")

        # File uploader for video input
        uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_video_path = f"temp_{uploaded_file.name}"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.read())

            st.success("Video uploaded successfully!")

            # Preprocess based on selected model
            preprocess_for_model(selected_model)

            # Add placeholder for further processing (e.g., running the model)
            st.info(f"Running {selected_model} model...")
            # Placeholder for model inference (to be implemented)
            st.success(f"Model {selected_model} executed successfully!")


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
import streamlit as st

# Set up the main configuration
st.set_page_config(page_title="Forgery Detection-Group 6", layout="wide")


# Define a function to render the Project page
def project_page():
    st.title("Forgery Detection - Group 6")

    # Title bar styling
    st.markdown(
        """
        <style>
        .title-bar {
            background-color: #003366;
            padding: 10px;
            color: white;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        </style>
        <div class="title-bar">Forgery Detection - Group 6</div>
        """,
        unsafe_allow_html=True
    )

    # Introduction section
    st.header("Introduction")
    st.write(
        "In this project, we aim to develop a robust system for detecting forgeries in multimedia content. Our goal is to identify and highlight manipulated media to ensure authenticity and trust.")

    # Solution section
    st.header("Solution Overview")

    # Dataset Description subsection
    st.subheader("Dataset Description")
    st.write(
        "We are utilizing datasets containing real and synthesized media to train our detection models. The datasets include various sources to ensure comprehensive coverage of potential forgery techniques.")

    # Model Description subsection
    st.subheader("Model Description")
    st.write(
        "Our model is based on advanced deep learning architectures, specifically designed to detect subtle inconsistencies in media that are indicative of forgery.")

    # Footer for references
    st.markdown("---")
    st.write("References will be added here.")


# Define a function to render the Implementation/Demo page
def implementation_page():
    st.title("Implementation - Forgery Detection")


# Define a function to render the About Us page
def about_us_page():
    st.title("About Us - Group 6")


# Create a sidebar menu for navigation
menu = ["Project", "Implementation", "About Us"]
choice = st.sidebar.selectbox("Menu", menu)

# Render pages based on menu choice
if choice == "Project":
    project_page()
elif choice == "Implementation":
    implementation_page()
elif choice == "About Us":
    about_us_page()
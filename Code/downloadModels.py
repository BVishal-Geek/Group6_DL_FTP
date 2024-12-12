import os
import gdown

# Paths to models folder and required models
models_folder = os.path.join(os.getcwd(), "..", "models")
inception_model_path = os.path.join(models_folder, "best_model.keras")
vgg_model_path = os.path.join(models_folder, "finetuned_vgg16.keras")

def download_google_drive_folder(output_dir):
    """
    Downloads a folder from Google Drive using its public shareable link.

    Args:
        output_dir (str): The local directory where the folder will be downloaded.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the URL for gdown to use
    download_url = f"https://drive.google.com/drive/folders/1UQA6rNWOu57Ae4VQltlkhOg3Bsw7Myz2?usp=sharing"

    print("Downloading folder from Google Drive...")
    gdown.download_folder(download_url, quiet=False, output=output_dir)

    print(f"Folder downloaded to: {output_dir}")


def check_and_download_models():
    """
    Checks if the models folder and required models exist.
    If not, downloads them using the download_google_drive_folder function.
    """
    # Check if the folder exists
    if os.path.exists(models_folder):
        # Check if both model files exist
        if os.path.exists(inception_model_path) and os.path.exists(vgg_model_path):
            print("All required models are already present. Skipping download.")
            return
        else:
            print("Some models are missing. Downloading models...")
    else:
        print("Models folder not found. Downloading models...")

    # Call the download function
    try:
        download_google_drive_folder(models_folder)
        print("Models downloaded successfully!")
    except Exception as e:
        print(f"Failed to download models: {e}")


# Main script
if __name__ == "__main__":
    # Check and download models
    check_and_download_models()

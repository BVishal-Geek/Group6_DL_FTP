import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, optimizers
from sklearn.metrics import accuracy_score, f1_score

# Paths to data
excel_file_path = "/home/ubuntu/Final_Project/data/frames_labels.xlsx"  # Path to labels Excel file
images_dir_path = "/home/ubuntu/Final_Project/data/frames/"  # Path to frames directory


# Step 1: Load Excel and Analyze Data
def analyze_data_from_excel(excel_file):
    # Load the Excel file
    df = pd.read_excel(excel_file)

    # Print total number of images
    total_images = len(df)
    print(f"Total number of images: {total_images}")

    # Count labels (0s and 1s)
    label_counts = df['label'].value_counts()
    num_zeros = label_counts.get(0, 0)
    num_ones = label_counts.get(1, 0)
    print(f"Number of 0s (Fake Images): {num_zeros}")
    print(f"Number of 1s (Real Images): {num_ones}")

    return df


# Step 2: Calculate Mean RGB Values for Real and Fake Images
def calculate_mean_rgb(df, images_dir, img_size=(64, 64)):
    real_images = []
    fake_images = []

    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, row['image_name'])

        # Load image and resize
        img = tf.keras.utils.load_img(image_path, target_size=img_size)
        img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize pixel values

        if row['label'] == 1:  # Real image
            real_images.append(img_array)
        else:  # Fake image
            fake_images.append(img_array)

    # Calculate mean RGB values for real and fake images
    mean_real_rgb = np.mean(np.array(real_images), axis=(0, 1, 2))
    mean_fake_rgb = np.mean(np.array(fake_images), axis=(0, 1, 2))

    print(f"Mean RGB values for Real Images: {mean_real_rgb}")
    print(f"Mean RGB values for Fake Images: {mean_fake_rgb}")

    return mean_real_rgb, mean_fake_rgb


# Step 3: Preprocess Images and Add Auxiliary Data
def preprocess_images_and_auxiliary_data(df, images_dir, mean_real_rgb, mean_fake_rgb, img_size=(64, 64)):
    images = []
    labels = []
    auxiliary_data = []

    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, row['image_name'])

        # Load image and resize
        img = tf.keras.utils.load_img(image_path, target_size=img_size)
        img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize pixel values

        # Use precomputed mean RGB values as auxiliary data
        if row['label'] == 1:  # Real image
            aux_feature = mean_real_rgb
        else:  # Fake image
            aux_feature = mean_fake_rgb

        images.append(img_array)
        labels.append(row['label'])
        auxiliary_data.append(aux_feature)

    return np.array(images), np.array(labels), np.array(auxiliary_data)


# Step 4: Build a Basic CNN Model with Auxiliary Features
def build_basic_cnn_with_auxiliary(input_shape):
    image_input = Input(shape=input_shape)

    # Basic CNN architecture
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x_flattened = layers.Flatten()(x)

    aux_input = Input(shape=(3,))

    combined_input = layers.concatenate([x_flattened, aux_input])

    x_combined = layers.Dense(128, activation='relu')(combined_input)

    final_output = layers.Dense(1, activation='sigmoid', dtype='float32')(x_combined)

    model = Model(inputs=[image_input, aux_input], outputs=final_output)

    return model


# Step 5: Focal Loss Implementation
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred_clipped = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy_loss = -y_true * tf.math.log(y_pred_clipped)
        weights = alpha * y_true + (1 - alpha) * (1 - y_true)
        focal_loss_value = weights * tf.pow((1 - y_pred_clipped), gamma) * cross_entropy_loss
        return tf.reduce_mean(focal_loss_value)

    return focal_loss_fixed


# Step 6: Train and Evaluate the Model
def train_and_evaluate_model(model,
                             X_train,
                             aux_train,
                             y_train,
                             X_val,
                             aux_val,
                             y_val,
                             X_test,
                             aux_test,
                             y_test,
                             batch_size=32,
                             epochs=20):
    model.compile(optimizer=optimizers.Adam(), loss=focal_loss(), metrics=['accuracy'])

    history = model.fit(
        [X_train, aux_train],
        y_train,
        validation_data=([X_val, aux_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate on test data
    predictions = model.predict([X_test, aux_test])
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test.flatten(), predicted_classes)
    f1_score_value = f1_score(y_test.flatten(), predicted_classes)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1_score_value:.4f}")


# Main Execution Flow
if __name__ == "__main__":
    # Analyze data from Excel file
    df_data = analyze_data_from_excel(excel_file_path)

    # Calculate mean RGB values for real and fake images
    mean_real_rgb, mean_fake_rgb = calculate_mean_rgb(df_data, images_dir_path)

    # Preprocess images and add auxiliary data
    X_images, y_labels, X_auxiliary_data = preprocess_images_and_auxiliary_data(
        df_data,
        images_dir_path,
        mean_real_rgb,
        mean_fake_rgb
    )

    # Split data into train/validation/test sets (e.g., 60/20/20 split)
    train_size = int(0.6 * len(X_images))
    val_size = int(0.2 * len(X_images))

    X_train = X_images[:train_size]
    y_train = y_labels[:train_size]

    X_val = X_images[train_size:train_size + val_size]
    y_val = y_labels[train_size:train_size + val_size]

    X_test = X_images[train_size + val_size:]
    y_test = y_labels[train_size + val_size:]

    aux_train = X_auxiliary_data[:train_size]
    aux_val = X_auxiliary_data[train_size:train_size + val_size]
    aux_test = X_auxiliary_data[train_size + val_size:]

    # Build basic CNN model with auxiliary inputs
    basic_cnn_model_with_auxiliary_inputs = build_basic_cnn_with_auxiliary((64, 64, 3))

    # Train and evaluate the model
    train_and_evaluate_model(
        basic_cnn_model_with_auxiliary_inputs,
        X_train,
        aux_train,
        y_train,
        X_val,
        aux_val,
        y_val,
        X_test,
        aux_test,
        y_test
    )
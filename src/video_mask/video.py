import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, optimizers
from sklearn.metrics import accuracy_score, f1_score

# Paths to data and model save location
excel_file_path = "/home/ubuntu/Final_Project/data/frames_labels.xlsx"
images_dir_path = "/home/ubuntu/Final_Project/data/frames/"
model_save_path = "/home/ubuntu/Final_Project/saved_model/fake_vs_real_model.h5"

# Corrected Mean RGB Values
mean_real_rgb = [0.31304556, 0.26466683, 0.26480529]
mean_fake_rgb = [0.33616986, 0.28414449, 0.27994417]


# Function to load all images and auxiliary data into memory
def load_all_images_and_auxiliary_data(df, images_dir, mean_real_rgb, mean_fake_rgb, img_size=(64, 64)):
    images = []
    labels = []
    auxiliary_data = []

    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, row['image_name'])
        img = tf.keras.utils.load_img(image_path, target_size=img_size)
        img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normalize pixel values

        # Assign auxiliary data based on label
        if row['label'] == 1:  # Real image
            aux_feature = mean_real_rgb
        else:  # Fake image
            aux_feature = mean_fake_rgb

        images.append(img_array)
        labels.append(row['label'])
        auxiliary_data.append(aux_feature)

    return np.array(images), np.array(labels), np.array(auxiliary_data)


# Build a Basic CNN Model with Auxiliary Features
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


# Focal Loss Implementation
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred_clipped = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy_loss = -y_true * tf.math.log(y_pred_clipped)
        weights = alpha * y_true + (1 - alpha) * (1 - y_true)
        focal_loss_value = weights * tf.pow((1 - y_pred_clipped), gamma) * cross_entropy_loss
        return tf.reduce_mean(focal_loss_value)

    return focal_loss_fixed


# Train and Evaluate the Model
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

    # Train the model
    history = model.fit(
        [X_train, aux_train],
        y_train,
        validation_data=([X_val, aux_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate([X_test, aux_test], y_test)

    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Predict on test data for F1 score calculation
    predictions = model.predict([X_test, aux_test])
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    f1_score_value = f1_score(y_test.flatten(), predicted_classes)

    print(f"Test F1 Score: {f1_score_value:.4f}")


# Main Execution Flow
if __name__ == "__main__":
    # Load dataset from Excel file
    df_data = pd.read_excel(excel_file_path)

    # Split data into train/validation/test sets (60%/20%/20%)
    train_split_idx = int(0.6 * len(df_data))
    val_split_idx = int(0.8 * len(df_data))

    train_df = df_data.iloc[:train_split_idx]
    val_df = df_data.iloc[train_split_idx:val_split_idx]
    test_df = df_data.iloc[val_split_idx:]

    print("Dataset split completed:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Testing samples: {len(test_df)}")

    # Load all images and auxiliary data into memory
    print("Loading training data...")
    X_train, y_train, aux_train = load_all_images_and_auxiliary_data(train_df, images_dir_path, mean_real_rgb,
                                                                     mean_fake_rgb)

    print("Loading validation data...")
    X_val, y_val, aux_val = load_all_images_and_auxiliary_data(val_df, images_dir_path, mean_real_rgb, mean_fake_rgb)

    print("Loading testing data...")
    X_test, y_test, aux_test = load_all_images_and_auxiliary_data(test_df, images_dir_path, mean_real_rgb,
                                                                  mean_fake_rgb)

    # Build basic CNN model with auxiliary inputs
    basic_cnn_model_with_auxiliary_inputs = build_basic_cnn_with_auxiliary((64, 64, 3))

    # Train and evaluate the model with preloaded data
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
        y_test,
        batch_size=32,
        epochs=20
    )
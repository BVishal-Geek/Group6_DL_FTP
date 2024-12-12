import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, Input, Model
from sklearn.metrics import accuracy_score, f1_score

# Base directories for datasets
base_dir = '/home/ubuntu/Final_Project/data/'
excel_dir = '/home/ubuntu/Final_Project/excel/'

# Image size for model input
IMG_SIZE = (224, 224)

# Batch size for data loading
BATCH_SIZE = 32


def calculate_average_pixel_value(image_path):
    """Calculate the average pixel value of an image."""
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, IMG_SIZE)
        return np.mean(image, axis=(0, 1))  # Average over width and height
    else:
        return np.zeros(3)

def process_auxiliary_features(aux_features):
    """Ensure auxiliary features are in a flat, consistent format."""
    # Flatten or aggregate nested arrays
    processed_features = [
        feature if feature.ndim == 1 else np.mean(feature, axis=0)
        for feature in aux_features
    ]
    # Convert all features to float32
    return np.array(processed_features, dtype=np.float32)

def create_data_generator_with_auxiliary(excel_file, image_folder):
    """Create a data generator that includes auxiliary features."""
    df = pd.read_excel(excel_file)

    # Calculate auxiliary features (average pixel value) for each image
    df['avg_pixel_value'] = df['imageid'].apply(lambda x: calculate_average_pixel_value(os.path.join(image_folder, x)))

    datagen = ImageDataGenerator(rescale=1. / 255)  # Normalize pixel values to [0, 1]

    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=image_folder,
        x_col='imageid',
        y_col='classification',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='raw',  # Use 'raw' for regression or binary classification without one-hot encoding
        shuffle=True,
        validate_filenames=False
    )

    return generator, process_auxiliary_features(df['avg_pixel_value'].values)


# Create data generators for train, validation, and test datasets with auxiliary features
train_generator_ffhq, train_aux_features_ffhq = create_data_generator_with_auxiliary(
    os.path.join(excel_dir, 'ffhq_train.xlsx'),
    os.path.join(base_dir, 'ffhq/train')
)
train_generator_faceapp, train_aux_features_faceapp = create_data_generator_with_auxiliary(
    os.path.join(excel_dir, 'faceapp_train.xlsx'),
    os.path.join(base_dir, 'faceapp/train')
)

val_generator_ffhq, val_aux_features_ffhq = create_data_generator_with_auxiliary(
    os.path.join(excel_dir, 'ffhq_validation.xlsx'),
    os.path.join(base_dir, 'ffhq/validation')
)
val_generator_faceapp, val_aux_features_faceapp = create_data_generator_with_auxiliary(
    os.path.join(excel_dir, 'faceapp_validation.xlsx'),
    os.path.join(base_dir, 'faceapp/validation')
)

test_generator_ffhq, test_aux_features_ffhq = create_data_generator_with_auxiliary(
    os.path.join(excel_dir, 'ffhq_test.xlsx'),
    os.path.join(base_dir, 'ffhq/test')
)
test_generator_faceapp, test_aux_features_faceapp = create_data_generator_with_auxiliary(
    os.path.join(excel_dir, 'faceapp_test.xlsx'),
    os.path.join(base_dir, 'faceapp/test')
)


# Define the VGG-like architecture with auxiliary input
def build_vgg_model_with_aux(input_shape):
    # Main image input
    image_input = Input(shape=input_shape)

    # VGG-style convolutional blocks
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(image_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Flatten and fully connected layer for main path
    flat_x = layers.Flatten()(x)

    # Auxiliary input for average pixel values
    aux_input = Input(shape=(3,))

    # Concatenate auxiliary input with main path output
    combined_input = layers.concatenate([flat_x, aux_input])

    # Fully connected layer after concatenation
    combined_output = layers.Dense(4096, activation='relu')(combined_input)
    combined_output = layers.Dense(4096, activation='relu')(combined_output)

    # Output layer with sigmoid for binary classification
    final_output = layers.Dense(1, activation='sigmoid')(combined_output)

    # Create model with two inputs and one output
    model = Model(inputs=[image_input, aux_input], outputs=[final_output])

    return model


# Build the VGG-like model with auxiliary inputs
model = build_vgg_model_with_aux((IMG_SIZE[0], IMG_SIZE[1], 3))

# Compile the model using binary cross-entropy loss and Adam optimizer
model.compile(optimizer=optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()


# Combine generators and auxiliary features from both datasets for training and validation

def combine_generators_and_features(generator_ffhq, generator_faceapp,aux_features_ffhq,aux_features_faceapp):
    images_ffhq = np.concatenate([generator_ffhq[i][0] for i in range(len(generator_ffhq))])
    labels_ffhq = np.concatenate([generator_ffhq[i][1] for i in range(len(generator_ffhq))])

    images_faceapp = np.concatenate([generator_faceapp[i][0] for i in range(len(generator_faceapp))])
    labels_faceapp = np.concatenate([generator_faceapp[i][1] for i in range(len(generator_faceapp))])

    combined_images = np.concatenate([images_ffhq,
                                      images_faceapp])
    combined_labels = np.concatenate([labels_ffhq,
                                      labels_faceapp])
    combined_aux = np.concatenate([aux_features_ffhq,
                                   aux_features_faceapp])

    return combined_images,combined_labels,combined_aux


train_combined_images,train_combined_labels,train_combined_aux = combine_generators_and_features(train_generator_ffhq,
                                                     train_generator_faceapp,
                                                     train_aux_features_ffhq,
                                                     train_aux_features_faceapp)


val_combined_images,val_combined_labels,val_combined_aux = combine_generators_and_features(val_generator_ffhq,
                                                   val_generator_faceapp,
                                                   val_aux_features_ffhq,
                                                   val_aux_features_faceapp)


history = model.fit(
    [train_combined_images,
     train_combined_aux],
    train_combined_labels,
    validation_data=([val_combined_images,
                      val_combined_aux], val_combined_labels),
    epochs=50,
)

# Evaluate the model on test data using combined datasets and auxiliary features

test_combined_images,test_combined_labels,test_combined_aux = combine_generators_and_features(test_generator_ffhq,
                                                    test_generator_faceapp,
                                                    test_aux_features_ffhq,
                                                    test_aux_features_faceapp)

predictions = model.predict([test_combined_images,
                             test_combined_aux])
predicted_classes = (predictions > 0.35).astype(int)[:, 0]

# Calculate accuracy and F1 score on test data

accuracy = accuracy_score(test_combined_labels, predicted_classes)
f1 = f1_score(test_combined_labels, predicted_classes)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
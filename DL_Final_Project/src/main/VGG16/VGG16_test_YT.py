import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf
import sys
sys.path.append("../../")

from components.image_classification import *
from components.image_preprocessing import *
#%%
print('----------INSTANTIATING IMAGE GENERATORS----------')

test_directory = '../../../data/YT_Frames_test_ela'
BATCH_SIZE = 64
generator = ImageDataGenerator()
test_generator = generator.flow_from_directory(
    directory=test_directory,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
    seed=6303
)

#%%
# ----- LOAD TRAINED MODEL AND GENERATE PREDICTIONS -----
@tf.keras.utils.register_keras_serializable()
def f1_macro(y_true, y_pred):
    """
    Compute F1 macro score as a custom metric.
    """
    y_pred = tf.round(y_pred)  # Convert predictions to 0 or 1
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    # Compute the mean F1 score across all classes
    f1_macro = tf.reduce_mean(f1)
    return f1_macro
print('----------LOADING FINE TUNED MODEL----------')

model = load_model('finetuned_vgg16_ela.keras', custom_objects={'f1_macro': f1_macro})
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','recall'])

steps = test_generator.n // BATCH_SIZE
print(f'----------TEST STEPS: {steps}----------')

#%%
# Predict probabilities for test data
predictions = model.predict(test_generator)
predicted_labels = np.round(predictions).flatten()  # Convert probabilities to binary (0 or 1)

# Extract true labels
true_labels = test_generator.classes  # True labels for the dataset
# Ensure the number of predictions matches the number of true labels
if len(predicted_labels) != len(true_labels):
    print(f"Warning: Mismatch in lengths. Predictions: {len(predicted_labels)}, True Labels: {len(true_labels)}")
# Class indices mapping (optional)
class_indices = test_generator.class_indices  # Dictionary mapping class names to integer labels

# Calculate F1-macro score
f1_macro = f1_score(true_labels, predicted_labels, average='macro')

# Print results
print(f"Test F1 Macro Score: {f1_macro * 100:.2f}%")
print(f"Test Accuracy: {np.mean(predicted_labels == true_labels) * 100:.2f}%")
print(classification_report(true_labels, predicted_labels))

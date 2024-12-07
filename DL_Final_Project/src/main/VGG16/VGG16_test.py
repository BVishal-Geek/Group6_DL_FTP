import numpy as np
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, f1_score
import sys
sys.path.append("../../")

from components.utils import *
#%%
print('----------INSTANTIATING IMAGE GENERATORS----------')

test_directory = '../../../data/Frames_test'
BATCH_SIZE = 32
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

model = load_model('finetuned_vgg16.h5')
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

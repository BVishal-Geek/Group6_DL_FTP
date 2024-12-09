#%%
import cv2
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
#%%
# Define Crop Center Helper Function
def crop_center_square(frame):
    """Crop the center square of the frame."""
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Define Image Classification Class
class Image_Classification:
    def __init__(self, video_fp, finetuned_model, model_name, frame_size, seq_length):
        self.frame_size = frame_size
        self.seq_length = seq_length
        self.model_name = model_name
        self.video_fp = video_fp
        self.finetuned_model = finetuned_model

    def get_frames(self):
        cap = cv2.VideoCapture(self.video_fp)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.seq_length, dtype=int)

        for i in tqdm.tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                print(f"Frame {i} not readable in {self.video_fp}. Skipping.")
                break
            if i in frame_indices:
                # Crop and resize the frame
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, self.frame_size)
                if self.model_name == 'VGG16':
                    frame = preprocess_input(frame)
                frames.append(frame)
        cap.release()

        frames = np.array(frames)
        if len(frames) < self.seq_length:
            print("Error: Video has fewer frames than the required sequence length.")

        return frames

    def get_prediction(self, frames):
        """
        Predict the class for a video by aggregating frame-level predictions.
        """
        predictions = []
        for frame in frames:  # Iterate over individual frames
            frame = np.expand_dims(frame, axis=0)  # Add batch dimension (1, 224, 224, 3)
            prediction = self.finetuned_model.predict(frame, verbose=0)  # Predict for single frame
            predictions.append(prediction[0][0])  # Extract the probability for the positive class

        # Aggregate predictions (e.g., take the average probability)
        avg_prediction = np.mean(predictions)
        label = int(avg_prediction > 0.5)  # Threshold at 0.5
        label_name = "Fake" if label == 1 else "Real"

        print(f"\n\n Prediction: {label_name} (Average Probability: {avg_prediction:.4f}) \n\n")
        return label_name, avg_prediction

#%%
# Define Custom F1-Macro Function
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

#%%
if __name__ == '__main__':
    print('----------LOADING FINE TUNED MODEL----------')
    model = load_model('finetuned_vgg16.keras', custom_objects={'f1_macro': f1_macro})

    print('----------COMPILING FINE TUNED MODEL----------')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','recall'])

    print('----------GENERATION VIDEO PREDICTION----------')
    video_fp = '../../data/Individual_Video/Fake_Video1.mp4'
    video_pred = Image_Classification(video_fp=video_fp,
                                      finetuned_model=model,
                                      model_name='VGG16',
                                      frame_size=(224,224),
                                      seq_length=10)
    frames = video_pred.get_frames()
    video_pred.get_prediction(frames)
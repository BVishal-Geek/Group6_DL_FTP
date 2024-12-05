#%%
import tensorflow as tf
from scipy.stats.tests.test_continuous_fit_censored import optimizer
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Rescaling, Dense
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from mtcnn import MTCNN


#%%

class BinaryImageClassifier:
    def __init__(self, model_type, data_dir, epochs=10, batch_size=10, IMG_SIZE=200):
        self.model_type = model_type
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.IMG_SIZE = IMG_SIZE
        self.history = None

    def load_data(self):
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="both",
            seed=123,
            image_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=32,
        )
        normalization = tf.keras.layers.Rescaling(1.0/255)
        train_ds = train_ds.map(lambda x, y: (normalization(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization(x), y))

        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = train_ds.prefetch(10)
        self.val_ds = val_ds.prefetch(10)

    def build_model(self):
        if self.model_type == "CNN":
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])

            self.model.compile(optimizer= "adam",
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train_model(self):
        if self.model is None:
            raise ValueError('Model not built')
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

    def evaluate_model(self):
        if self.history is None:
            raise ValueError("Model has not been trained. Call train_model() first.")

        val_loss, val_accuracy = self.model.evaluate(self.val_ds)
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {val_accuracy}")

        # Generate predictions
        val_predictions = (self.model.predict(self.val_ds.map(lambda x, y: x)).ravel() > 0.5).astype("int32")
        val_labels_flat = tf.concat([y for _, y in self.val_ds], axis=0).numpy()

        # Classification metrics
        print("\nClassification Report:")
        print(classification_report(val_labels_flat, val_predictions, target_names=["Real", "Fake"]))

        print("\nConfusion Matrix:")
        print(confusion_matrix(val_labels_flat, val_predictions))

        # AUC-ROC score
        val_probabilities = self.model.predict(self.val_ds.map(lambda x, y: x)).ravel()
        auc_roc = roc_auc_score(val_labels_flat, val_probabilities)
        print(f"\nAUC-ROC Score: {auc_roc:.4f}")

    def save_model(self, model_path="binary_classifier_model.h5"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        self.model.save(model_path)
        print(f"Model saved to {model_path}.")

classifier = BinaryImageClassifier(
    model_type="CNN",
    data_dir='/home/ubuntu/Group6_DL_FTP/data/Images',
    IMG_SIZE=100,
    batch_size=32,
    epochs=10
)

classifier.load_data()
classifier.build_model()
classifier.train_model()
classifier.evaluate_model()
classifier.save_model()

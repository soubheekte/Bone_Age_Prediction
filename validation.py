import os
from config.config import DataConfig, Config
import tensorflow as tf
import numpy as np
from model import BoneAgeDataGenerator

class ValidationModel():

    def __init__(self):
        pass



    def validate_model(self):
        """
        Validates the given model on the provided validation data.

        Args:
            model: The trained model to be validated.
            validation_data: A tuple (X_val, y_val) containing validation features and labels.

        Returns:
            float: The validation accuracy of the model.
        """
        model_save_dir = os.path.join(Config.BASEDIR, 'model_files')

        # Find the latest .keras model; fallback to latest .h5; otherwise keep directory fallback
        latest_model_path = None
        if os.path.isdir(model_save_dir):
            try:
                files = os.listdir(model_save_dir)
            except Exception:
                files = []

            keras_files = [os.path.join(model_save_dir, f) for f in files if f.endswith('.keras')]
            h5_files = [os.path.join(model_save_dir, f) for f in files if f.endswith('.h5')]

            if keras_files:
                latest_model_path = max(keras_files, key=os.path.getmtime)
            elif h5_files:
                latest_model_path = max(h5_files, key=os.path.getmtime)

        # Load chosen model (file or directory) with helpful error if it fails
        try:
            if latest_model_path:
                loaded_model = tf.keras.models.load_model(latest_model_path, custom_objects={'mae': tf.keras.metrics.MeanAbsoluteError()})
            else:
                loaded_model = tf.keras.models.load_model(model_save_dir, custom_objects={'mae': tf.keras.metrics.MeanAbsoluteError()})
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{latest_model_path or model_save_dir}': {e}") from e
        

        # Preprocess validation data
        

        # Predict on the validation data
        val_generator = BoneAgeDataGenerator(DataConfig.valid_processed_images_dir, DataConfig.valid_processed_labels_dir, batch_size=1)
        predictions = loaded_model.predict(val_generator)

        # Prepare list of validation label files for sample printing
        val_label_files = []
        if os.path.isdir(DataConfig.valid_processed_labels_dir):
            try:
                val_label_files = sorted([
                    os.path.join(DataConfig.valid_processed_labels_dir, f)
                    for f in os.listdir(DataConfig.valid_processed_labels_dir)
                    if f.endswith('.npy')
                ])
            except Exception:
                val_label_files = []

        # Display the first few predictions and actual values
        print("Sample Predictions vs Actuals:")
        for i in range(min(10, len(val_label_files))):  # Display up to 10 samples
            actual_labels = np.load(val_label_files[i])
            print(f"Batch {i+1}:")
            for j in range(min(5, len(actual_labels))):  # Display up to 5 items per batch
                print(f"  Actual: {actual_labels[j][0]:.2f}, Predicted: {predictions[i * val_generator.batch_size + j][0]:.2f}")
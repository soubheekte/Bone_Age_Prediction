import os
from config.config import DataConfig, Config
import tensorflow as tf
import numpy as np
from model import BoneAgeDataGenerator
from utils.common_utils import compute_regression_metrics
from utils.logger import get_logger

class ValidationModel():

    def __init__(self):
        self.image_batch_files = sorted([os.path.join(DataConfig.valid_processed_images_dir, f) for f in os.listdir(DataConfig.valid_processed_images_dir) if f.endswith('.npy')])
        self.gender_files = sorted([os.path.join(DataConfig.valid_processed_genders_dir, f) for f in os.listdir(DataConfig.valid_processed_genders_dir) if f.endswith('.npy')])
        self.label_batch_files = sorted([os.path.join(DataConfig.valid_processed_labels_dir, f) for f in os.listdir(DataConfig.valid_processed_labels_dir) if f.endswith('.npy')])
        # initialize custom logger
        self.logger = get_logger(__name__)

    # Replace your existing validate_model function with this one

    def validate_model(self):
        """
        Validates the given model on the provided validation data.
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
            self.logger.info("Loaded model from %s", latest_model_path or model_save_dir)
        except Exception as e:
            # log the exception and re-raise for visibility
            self.logger.exception("Failed to load model from '%s': %s", latest_model_path or model_save_dir, e)
            raise RuntimeError(f"Failed to load model from '{latest_model_path or model_save_dir}': {e}") from e
        
        # Predict on the validation data using the generator
        val_generator = BoneAgeDataGenerator(self.image_batch_files, self.gender_files, self.label_batch_files)
        predictions = loaded_model.predict(val_generator)

        labels_flat = np.concatenate([np.load(f).ravel() for f in self.label_batch_files])
        labels_flat = labels_flat.astype(int)


        # Compute regression metrics
        metrics = compute_regression_metrics(labels_flat, predictions.ravel())

        # log the metrics using the custom logger
        for metric_name, metric_value in metrics.items():
            self.logger.info("%s: %.4f", metric_name, metric_value)
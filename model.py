import numpy as np
import os
# from tensorflow.keras.utils import Sequence
import tensorflow as tf
# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
from config.config import DataConfig, Config
from utils.logger import get_logger
from utils.common_utils import compute_regression_metrics, save_model_with_rotation
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model


class BoneAgeDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras Sequence to load preprocessed image and label batches from .npy files.
    """
    def __init__(self, image_files, gender_files, label_files):
        """
        Args:
            image_files (list): List of paths to the preprocessed image .npy files.
            gender_files (list): List of paths to the preprocessed gender .npy files.
            label_files (list): List of paths to the preprocessed label .npy files.
            batch_size (int): The batch size for training.
        """
        self.image_files = image_files
        self.gender_files = gender_files
        self.label_files = label_files
        self.indexes = np.arange(len(self.image_files)) # Use file index as the basis for shuffling

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Generates one batch of data.
        """
        # Get the file index for this batch
        file_index = self.indexes[index]

        # Load the image and label data for this file
        batch_images = np.load(self.image_files[file_index])
        batch_gender_data = np.load(self.gender_files[file_index])
        batch_labels = np.load(self.label_files[file_index])

        # Since each file is a batch, we just return the loaded data
        return (batch_images, batch_gender_data), batch_labels

    def on_epoch_end(self):
        """
        Called at the end of each epoch. Shuffles the file indexes.
        """
        np.random.shuffle(self.indexes)



class BoneAgeModelTrainer():

    def __init__(self):

        # create a logger for the trainer early so we can log device detection
        self.logger = get_logger(__name__)

        # configure TF device behavior (enable memory growth when GPUs present)
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception as e:
                        # best-effort; continue if this fails
                        self.logger.warning("Could not set memory growth for GPU %s: %s", gpu, e)
                gpu_names = [g.name for g in gpus]
                self.logger.info("GPUs detected and memory growth enabled: %s", gpu_names)
            else:
                self.logger.info("No GPU devices detected. Training will run on CPU.")
        except Exception as e:
            # if anything goes wrong during device detection, log and continue
            self.logger.exception("Failed to detect/configure GPU devices: %s", e)

        # load the preprocessed data file paths
        self.image_batch_files = sorted([os.path.join(DataConfig.train_processed_images_dir, f) for f in os.listdir(DataConfig.train_processed_images_dir) if f.endswith('.npy')])
        self.male_data_files = sorted([os.path.join(DataConfig.train_processed_genders_dir, f) for f in os.listdir(DataConfig.train_processed_genders_dir) if f.endswith('.npy')])
        self.label_batch_files = sorted([os.path.join(DataConfig.train_processed_labels_dir, f) for f in os.listdir(DataConfig.train_processed_labels_dir) if f.endswith('.npy')])


        # create a logger for the trainer
        
        self.logger.info("BoneAgeModelTrainer initialized.")
        self.logger.info("Found %d image batch files, %d gender files, %d label files",
                         len(self.image_batch_files), len(self.male_data_files), len(self.label_batch_files))


    def model_definition(self):
        # 1. Define the input layers for each data source
        image_input = tf.keras.Input(shape=(224, 224, 1), name='image_input')
        male_input = tf.keras.Input(shape=(1,), name='male_input')

        # 2. Build the CNN branch for the image input
        x = tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu')(image_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        cnn_output = tf.keras.layers.Dense(8, activation='linear')(x) # Reduced for simplicity before merge

        # 3. Build the dense branch for the numerical ('male') input
        y = tf.keras.layers.Dense(16, activation='relu')(male_input)
        y = tf.keras.layers.Dense(8, activation='relu')(y)

        # 4. Concatenate the output of both branches
        combined_features = tf.keras.layers.concatenate([cnn_output, y])

        # 5. Add final dense layers for regression
        z = tf.keras.layers.Dense(4096, activation='relu')(combined_features)
        z = tf.keras.layers.Dropout(0.5)(z)
        z = tf.keras.layers.Dense(4096, activation='relu')(z)
        z = tf.keras.layers.Dropout(0.5)(z)
        final_output = tf.keras.layers.Dense(1, activation='linear')(z)

        # 6. Define the final model with two inputs and one output
        self.model = tf.keras.models.Model(inputs=[image_input, male_input], outputs=final_output)


        self.logger.info("Model architecture defined. Summary follows:")
        try:
            # You can print the summary to verify the architecture
            self.model.summary()
        except Exception as e:
            self.logger.error("Error occurred while printing model summary: %s", e)

    def train_model(self):

        split_ratio = 1 - Config.TEST_SIZE      # % for training
        split_index = int(len(self.image_batch_files) * split_ratio)


        train_image_files = self.image_batch_files[:split_index]
        train_gender_files = self.male_data_files[:split_index]
        train_label_files = self.label_batch_files[:split_index]

        val_image_files = self.image_batch_files[split_index:]
        val_gender_files = self.male_data_files[split_index:]
        val_label_files = self.label_batch_files[split_index:]


        train_generator = BoneAgeDataGenerator(train_image_files, train_gender_files, train_label_files)
        val_generator = BoneAgeDataGenerator(val_image_files, val_gender_files, val_label_files)


        print(f"Number of training batch files: {len(train_image_files)}")
        print(f"Number of validation batch files: {len(val_image_files)}")


        # Using Adam optimizer and Mean Absolute Error as the loss function
        self.model.compile(optimizer='adam', loss='mae', metrics=['mae'])


        # 1. Instantiate the EarlyStopping callback
        # We will monitor the validation loss.
        # Patience is set to 5, so training will stop if val_loss doesn't improve for 5 straight epochs.
        # We will restore the weights from the best epoch.
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )

        # instantiate training progress logger callback
        training_logger_cb = TrainingLoggerCallback(self.logger, batch_log_freq=50)

        # Train the model
        # We will use the data generators created earlier
        try:
            history = self.model.fit(
                train_generator,
                epochs=Config.EPOCHS, # You can adjust the number of epochs
                validation_data=val_generator,
                callbacks=[early_stopping_callback, training_logger_cb] # Add checkpoint callback here
            )
        except Exception as e:
            self.logger.error("Error occurred during model training: %s", e)

        # use the new save helper which enforces max model files
        try:
            save_model_with_rotation(model=self.model, base_name='alexnet_bone_age_model')
        except Exception as e:
            self.logger.error("Error occurred while saving model: %s", e)

        # compute and log validation metrics after training completes
        try:
            if len(val_label_files) > 0:
                self.logger.info("Computing validation metrics on %d batch files", len(val_label_files))
                # model.predict supports the generator that yields ((images, genders), labels)
                preds = self.model.predict(val_generator)
                # flatten and load true labels from the .npy batch files
                labels_flat = np.concatenate([np.load(f).ravel() for f in val_label_files])
                # ensure shapes align
                preds_flat = preds.ravel()
                metrics = compute_regression_metrics(labels_flat.astype(int), preds_flat)
                for metric_name, metric_value in metrics.items():
                    self.logger.info("%s: %.4f", metric_name, metric_value)
            else:
                self.logger.warning("No validation label files found; skipping metric computation.")
        except Exception as e:
            self.logger.exception("Failed to compute validation metrics: %s", e)


# add this new callback class (place after imports or after BoneAgeDataGenerator)
class TrainingLoggerCallback(tf.keras.callbacks.Callback):
    """
    Logs training progress using provided logger.
    - Logs batch-level loss every `batch_log_freq` batches.
    - Logs epoch-level metrics at the end of each epoch.
    """
    def __init__(self, logger, batch_log_freq=50):
        super().__init__()
        self.logger = logger
        self.batch_log_freq = batch_log_freq

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        if (batch + 1) % self.batch_log_freq == 0:
            loss = logs.get('loss')
            self.logger.info("Batch %d: loss=%.4f", batch + 1, loss if loss is not None else float('nan'))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # log main metrics and validation metrics if present
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        metrics_summary = []
        if loss is not None:
            metrics_summary.append(f"loss={loss:.4f}")
        if val_loss is not None:
            metrics_summary.append(f"val_loss={val_loss:.4f}")
        # include any other metrics (e.g., mae, val_mae)
        for k, v in logs.items():
            if k not in ('loss', 'val_loss'):
                try:
                    metrics_summary.append(f"{k}={v:.4f}")
                except Exception:
                    metrics_summary.append(f"{k}={v}")
        self.logger.info("Epoch %d ended. %s", epoch + 1, ", ".join(metrics_summary))














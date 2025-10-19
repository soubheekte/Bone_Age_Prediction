import numpy as np
import os
# from tensorflow.keras.utils import Sequence
import tensorflow as tf
from config.config import DataConfig, Config
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model


class BoneAgeDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras Sequence to load preprocessed image and label batches from .npy files.
    """
    def __init__(self, image_files, label_files, batch_size):
        """
        Args:
            image_files (list): List of paths to the preprocessed image .npy files.
            label_files (list): List of paths to the preprocessed label .npy files.
            batch_size (int): The batch size for training.
        """
        self.image_files = image_files
        self.label_files = label_files
        self.batch_size = batch_size
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
        batch_labels = np.load(self.label_files[file_index])

        # Since each file is a batch, we just return the loaded data
        return batch_images, batch_labels

    def on_epoch_end(self):
        """
        Called at the end of each epoch. Shuffles the file indexes.
        """
        np.random.shuffle(self.indexes)



class BoneAgeModelTrainer():

    def __init__(self):

        self.image_batch_files = sorted([os.path.join(DataConfig.train_processed_images_dir, f) for f in os.listdir(DataConfig.train_processed_images_dir) if f.endswith('.npy')])
        self.label_batch_files = sorted([os.path.join(DataConfig.train_processed_labels_dir, f) for f in os.listdir(DataConfig.train_processed_labels_dir) if f.endswith('.npy')])


    def model_Defination(self):

        # Define a simplified AlexNet-like model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='linear') # Output layer for regression
    ])

    def train_model(self):

        split_ratio = 1 - Config.TEST_SIZE      # % for training
        split_index = int(len(self.image_batch_files) * split_ratio)


        train_image_files = self.image_batch_files[:split_index]
        train_label_files = self.label_batch_files[:split_index]

        val_image_files = self.image_batch_files[split_index:]
        val_label_files = self.label_batch_files[split_index:]


        train_generator = BoneAgeDataGenerator(train_image_files, train_label_files, batch_size=1)
        val_generator = BoneAgeDataGenerator(val_image_files, val_label_files, batch_size=1)
        

        print(f"Number of training batch files: {len(train_image_files)}")
        print(f"Number of validation batch files: {len(val_image_files)}")


        # Using Adam optimizer and Mean Absolute Error as the loss function
        self.model.compile(optimizer='adam', loss='mae', metrics=['mae'])


        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=5, restore_best_weights=True)

        # Train the model
        # We will use the data generators created earlier
        history = self.model.fit(
            train_generator,
            epochs=3, # You can adjust the number of epochs
            validation_data=val_generator,
            callbacks=[early_stopping] # Add the early stopping callback here
        )


        # Define the base directory
        model_save_dir = os.path.join(Config.BASEDIR, 'model_files')

        # Create the directory if it doesn't exist
        os.makedirs(model_save_dir, exist_ok=True)

        # Define a versioning scheme for the model filename
        # You could use a timestamp or a simple counter
        version = 1 # Starting version
        model_filename = f'alexnet_bone_age_model_v{version}.keras'
        model_save_path = os.path.join(model_save_dir, model_filename)

        # Check if the file already exists and increment the version if needed
        while os.path.exists(model_save_path):
            version += 1
            model_filename = f'alexnet_bone_age_model_v{version}.keras'
            model_save_path = os.path.join(model_save_dir, model_filename)

        # Save the trained model
        self.model.save(model_save_path)
        print(f"Model saved successfully to: {model_save_path}")
    















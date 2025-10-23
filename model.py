import numpy as np
import os
# from tensorflow.keras.utils import Sequence
import tensorflow as tf
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

        self.image_batch_files = sorted([os.path.join(DataConfig.train_processed_images_dir, f) for f in os.listdir(DataConfig.train_processed_images_dir) if f.endswith('.npy')])
        self.male_data_files = sorted([os.path.join(DataConfig.train_processed_genders_dir, f) for f in os.listdir(DataConfig.train_processed_genders_dir) if f.endswith('.npy')])
        self.label_batch_files = sorted([os.path.join(DataConfig.train_processed_labels_dir, f) for f in os.listdir(DataConfig.train_processed_labels_dir) if f.endswith('.npy')])



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

        # You can print the summary to verify the architecture
        self.model.summary()



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

        # Train the model
        # We will use the data generators created earlier
        history = self.model.fit(
            train_generator,
            epochs=Config.EPOCHS, # You can adjust the number of epochs
            validation_data=val_generator,
            callbacks=[early_stopping_callback] # Add the early stopping callback here
        )


        # use the new save helper which enforces max model files
        save_model_with_rotation(model=self.model, base_name='alexnet_bone_age_model')
















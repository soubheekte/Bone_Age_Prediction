import numpy as np
import pandas as pd
import polars as pl
import os
# import keras
from PIL import Image
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from config.config import DataConfig



class BoneAgeDataPreprocessing():

    def __init__(self, validation= False):

        if validation == False:
            self.images_dir = DataConfig.train_image_path
            self.labels_csv_path = DataConfig.labels_csv_path
            self.label_data = None
            self.label_load_preprocess()

            self.processed_images_dir = DataConfig.train_processed_images_dir
            self.processed_labels_dir = DataConfig.train_processed_labels_dir
            self.processed_genders_dir = DataConfig.train_processed_genders_dir
            self.processed_ids_file = DataConfig.train_processed_ids_file

            # # Create directories if they don't exist
            os.makedirs(self.processed_images_dir, exist_ok=True)
            os.makedirs(self.processed_labels_dir, exist_ok=True)
            os.makedirs(self.processed_genders_dir, exist_ok=True)

        elif validation == True:
            self.images_dir = DataConfig.valid_image_path
            self.labels_csv_path = DataConfig.labels_csv_path
            self.label_data = None
            self.label_load_preprocess()

            self.processed_images_dir = DataConfig.valid_processed_images_dir
            self.processed_labels_dir = DataConfig.valid_processed_labels_dir
            self.processed_genders_dir = DataConfig.valid_processed_genders_dir
            self.processed_ids_file = DataConfig.valid_processed_ids_file

            # # Create directories if they don't exist
            os.makedirs(self.processed_images_dir, exist_ok=True)
            os.makedirs(self.processed_labels_dir, exist_ok=True)
            os.makedirs(self.processed_genders_dir, exist_ok=True)



    def label_load_preprocess(self):
        self.label_data = pd.read_csv(self.labels_csv_path)
        # Create image paths from image IDs in the label dataframe
        self.label_data['image_path'] = self.label_data['id'].apply(lambda x: os.path.join(self.images_dir, f'{x}.png'))
        # Check if all image paths exist and remove all which are not there.
        self.label_data = self.label_data[self.label_data['image_path'].apply(lambda x: os.path.exists(x))]


    def load_and_preprocess_image(self, image_path):
        """
        Loads an X-ray image, resizes it, normalizes pixel values, and adds a batch dimension.

        Args:
            image_path (str): The file path to the image.

        Returns:
            np.ndarray: A numpy array of the preprocessed image with a batch dimension (1, height, width, channels).
        """
        # Load the X-ray image
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224), color_mode='grayscale')

        # Convert to numpy array
        img_array = tf.keras.utils.img_to_array(img)

        # Normalize pixel values to 0-1 range based on the maximum pixel value
        max_pixel_value = np.max(img_array)
        if max_pixel_value > 1:
            img_array = img_array / max_pixel_value

        # Add batch dimension (required by the model)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array


    def process_image_batch(self, batch_size=200):
        """
        Processes a batch of unprocessed images, saves them, and updates the progress file.

        Args:
            batch_size (int): The number of images to process in this batch.
        Returns:
            bool: True if a batch was processed, False otherwise.
        """
        # Load previously processed image IDs
        processed_ids = set()
        if os.path.exists(self.processed_ids_file):
            with open(self.processed_ids_file, 'r') as f:
                processed_ids = set(line.strip() for line in f)

        # Filter out already processed images from the label DataFrame
        unprocessed_label = self.label_data[~self.label_data['id'].astype(str).isin(processed_ids)].reset_index(drop=True)

        if len(unprocessed_label) == 0:
            print("All images preprocessing complete.")
            # Ask the user if they want to delete preprocessed data and restart
            restart = input("All images preprocessed. Do you want to delete preprocessed data and restart? (yes/no): ").lower()
            if restart == 'yes':
                print("Deleting preprocessed data and restarting...")
                # Delete preprocessed image and label files
                for f in os.listdir(self.processed_images_dir):
                    os.remove(os.path.join(self.processed_images_dir, f))
                for f in os.listdir(self.processed_labels_dir):
                    os.remove(os.path.join(self.processed_labels_dir, f))
                # Delete gender files if present
                if os.path.exists(self.processed_genders_dir):
                    for f in os.listdir(self.processed_genders_dir):
                        os.remove(os.path.join(self.processed_genders_dir, f))
                # Empty the processed IDs file
                if os.path.exists(self.processed_ids_file):
                    open(self.processed_ids_file, 'w').close()
                return True  # Indicate that processing should restart by returning True
            else:
                print("Keeping preprocessed data.")
                return False # Indicate that processing is complete

        # Select a batch of unprocessed images
        batch_label = unprocessed_label.head(batch_size).copy()

        # Preprocess the batch of images
        print(f"Preprocessing {len(batch_label)} images...")
        batch_preprocessed_images = np.array([self.load_and_preprocess_image(path).squeeze(axis=0) for path in batch_label['image_path']])
        batch_bone_ages = batch_label['boneage'].values
        # Extract male column for the batch and save separately
        batch_male = batch_label['male'].values

        # Generate unique filenames for saving
        batch_id = len(processed_ids) // batch_size + 1
        images_filename = os.path.join(self.processed_images_dir, f'batch_{batch_id}_images.npy')
        labels_filename = os.path.join(self.processed_labels_dir, f'batch_{batch_id}_labels.npy')

        genders_filename = os.path.join(self.processed_genders_dir, f'batch_{batch_id}_genders.npy')
        # Save the preprocessed batch data
        np.save(images_filename, batch_preprocessed_images)
        np.save(labels_filename, batch_bone_ages)
        np.save(genders_filename, batch_male)

        # Update the set of processed IDs
        processed_ids.update(batch_label['id'].astype(str).tolist())

        # Save the updated list of processed IDs
        with open(self.processed_ids_file, 'w') as f:
            for img_id in processed_ids:
                f.write(f"{img_id}\n")

        print(f"Processed and saved batch {batch_id} with {len(batch_label)} images.")
        print(f"Saved genders for batch {batch_id} to {self.processed_genders_dir}")
        print(f"Total images processed so far: {len(processed_ids)}")
        print(f"Images remaining to process: {len(unprocessed_label) - len(batch_label)}")

        return True # Indicate that a batch was processed

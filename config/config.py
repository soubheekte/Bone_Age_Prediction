
from dataclasses import dataclass


@dataclass
class Config:
    DEBUG: bool = True
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    IMAGE_SIZE: tuple = (256, 256)
    BATCH_SIZE: int = 2
    EPOCHS: int = 2
    LEARNING_RATE: float = 0.001
    BASEDIR: str = r"."

@dataclass
class DataConfig:
    train_image_path: str = r"Data/train_source_data"
    valid_image_path: str = r"Data/valid_source_data"
    labels_csv_path: str = r"Data/boneage-dataset-labels.csv"

    train_processed_images_dir: str = r"Data/train_preprocessed_images"
    train_processed_labels_dir: str = r"Data/train_preprocessed_labels"
    train_processed_ids_file: str = r"Data/train_processed_image_ids.txt"
    train_processed_genders_dir: str = r"Data/train_preprocessed_genders"

    valid_processed_images_dir: str = r"Data/valid_preprocessed_images"
    valid_processed_labels_dir: str = r"Data/valid_preprocessed_labels"
    valid_processed_ids_file: str = r"Data/valid_processed_image_ids.txt"
    valid_processed_genders_dir: str = r"Data/valid_preprocessed_genders"

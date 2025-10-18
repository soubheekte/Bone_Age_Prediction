
from dataclasses import dataclass


@dataclass
class Config:
    DEBUG: bool = True
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    IMAGE_SIZE: tuple = (256, 256)
    BATCH_SIZE: int = 2
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    BASEDIR: str = r"."
    MODEL_SAVE_PATH: str = r"models/bone_age_model.h5"
    LOG_FILE_PATH: str = r"logs/pipeline.log"

@dataclass
class DataConfig:
    image_path: str = r"Data/small_data"
    labels_csv_path: str = r"Data/boneage-dataset-labels.csv"

    processed_images_dir: str = r"Data/preprocessed_images"
    processed_labels_dir: str = r"Data/preprocessed_labels"
    processed_ids_file: str = r"Data/processed_image_ids.txt"


from dataclasses import dataclass


@dataclass
class DataConfig:
    image_path: str = r"Data/small_data"
    labels_csv_path: str = r"Data/boneage-dataset-labels.csv"

    processed_images_dir: str = r"Data/preprocessed_images"
    processed_labels_dir: str = r"Data/preprocessed_labels"
    processed_ids_file: str = r"Data/processed_image_ids.txt"

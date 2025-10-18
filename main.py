from preprocessing_images import *
from utils.logger import get_logger
from config.config import Config, DataConfig

if __name__ == "__main__":

    logger = get_logger(__name__)
    logger.info("Logger is configured and ready to use.")

    data_preprocessor = BoneAgeDataPreprocessing()

    while (data_preprocessor.process_image_batch(Config.BATCH_SIZE)):
        logger.info(f"Processed a batch of images with batch size {Config.BATCH_SIZE}.")







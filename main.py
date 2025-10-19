from model import BoneAgeModelTrainer
from preprocessing_images import BoneAgeDataPreprocessing
from utils.logger import get_logger
from config.config import Config, DataConfig
import argparse

parser = argparse.ArgumentParser(description="Bone Age Prediction")
parser.add_argument("--preprocess",
                    choices=["True", "False"],
                    default="False", 
                    help="Enable preprocessing step")


parser.add_argument("--train",
                    choices=["True", "False"],
                    default="False", 
                    help="Enable training step")

args = parser.parse_args()


if __name__ == "__main__":

    logger = get_logger()
    logger.info("Logger is configured and ready to use.")

    if args.preprocess == "True":
        data_preprocessor = BoneAgeDataPreprocessing()

        while (data_preprocessor.process_image_batch(Config.BATCH_SIZE)):
            logger.info(f"Processed a batch of images with batch size {Config.BATCH_SIZE}.")

    if args.train == "True":
        logger.info("Training step is enabled.")

        model_trainer = BoneAgeModelTrainer()
        model_trainer.model_Defination()
        model_trainer.train_model()





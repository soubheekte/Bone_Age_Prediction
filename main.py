from model import BoneAgeModelTrainer
from preprocessing_images import BoneAgeDataPreprocessing
from utils.logger import get_logger
from config.config import Config, DataConfig
import argparse
from validation import ValidationModel
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description="Bone Age Prediction")
parser.add_argument("--preprocess",
                    choices=["train", "validate", "both", "False"],
                    default="train", 
                    help="Enable preprocessing step")


parser.add_argument("--train",
                    choices=["True", "False"],
                    default="False", 
                    help="Enable training step")

parser.add_argument("--validate",
                    choices=["True", "False"],
                    default="False", 
                    help="Enable validation step")


parser.add_argument("--run_pipeline",
                    choices=["True", "False"],
                    default="False", 
                    help="Enable entire pipeline: preprocessing, training, validation")

args = parser.parse_args()


if __name__ == "__main__":

    logger = get_logger()
    logger.info("Logger is configured and ready to use.")

    if args.preprocess.lower() == "train" or args.preprocess.lower() == "both" or args.run_pipeline.lower() == "true":
        data_preprocessor = BoneAgeDataPreprocessing()

        while (data_preprocessor.process_image_batch(Config.BATCH_SIZE)):
            logger.info(f"Processed a batch of images with batch size {Config.BATCH_SIZE}.")

    if args.preprocess.lower() == "validate" or args.preprocess.lower() == "both" or args.run_pipeline.lower() == "true":
        data_preprocessor = BoneAgeDataPreprocessing(validation=True)

        while (data_preprocessor.process_image_batch(Config.BATCH_SIZE)):
            logger.info(f"Processed a batch of validation images with batch size {Config.BATCH_SIZE}.")

    if args.train.lower() == "true" or args.run_pipeline.lower() == "true":
        logger.info("Training step is enabled.")

        model_trainer = BoneAgeModelTrainer()
        model_trainer.model_definition()
        model_trainer.train_model()

    if args.validate.lower() == "true" or args.run_pipeline.lower() == "true":
        logger.info("Validation step is enabled.")

        validator = ValidationModel()
        validator.validate_model()

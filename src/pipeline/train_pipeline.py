import os
import sys
import pandas as pd

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

from src.exception import CustomException
from src.logger import logging



def main():
    try:
        logging.info("Starting training pipeline...")

        # Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train data path: {train_path}, Test data path: {test_path}")

        # Data Preprocessing
        data_transformation = DataTransformation()
        train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_path, test_path)
        logging.info("Data Preprocessing completed.")

        # Model Training
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model Training completed. R2 Score: {r2_score}")

    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
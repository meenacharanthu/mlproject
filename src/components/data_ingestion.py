import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation, DataTransformationConfig


@dataclass
class DataIngestionConfig():
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw_data.csv')
    

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Data Ingestion Started')
            df = pd.read_csv('notebook\data\StudentsPerformance.csv')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)


            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)
            logging.info('Raw Data Saved')

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)
            logging.info('Train and Test Data Saved')
            logging.info('Data Ingestion Completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f'Error in Data Ingestion: {str(e)}')
            raise CustomException(e,sys)

if __name__ == '__main__':

    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    # print(train_path, test_path)
    logging.info('Data Ingestion Completed and saved artifacts')

    obj1 = DataTransformation()
    train_arr, test_arr,_ = obj1.initiate_data_transformation(train_path, test_path)
    logging.info('Data Transformation Completed and saved preprocessor object')

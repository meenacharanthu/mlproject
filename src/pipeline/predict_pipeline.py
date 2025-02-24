import sys
import os
import pandas as pd
# import logging

from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj

# logging.basicConfig(level=logging.DEBUG)

class Predict_pipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            logging.info('Loading model')
            model = load_obj(model_path)
            logging.info('Model loaded, now loading preprocessor')
            preprocessor = load_obj(preprocessor_path)
            logging.info('Preprocessor loaded')
            
            logging.info('Transforming input features')
            data_scaled = preprocessor.transform(features)
            logging.info('Making prediction')
            preds = model.predict(data_scaled)
            logging.info(f'Prediction: {preds}')
            return preds

        except Exception as e:
            logging.info(f"Error in predict method: {e}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education, lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_df(self):
        try:
            input_dict = {
                'gender': [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            logging.info("Creating DataFrame from input data")
            df = pd.DataFrame(input_dict)
            logging.info(f"DataFrame created: {df}")
            return df
        except Exception as e:
            logging.info(f"Error in get_data_as_df method: {e}")
            raise CustomException(e, sys)

# if __name__ == "__main__":
    # sample_data = CustomData(
    #     gender="female",
    #     race_ethnicity="group B",
    #     parental_level_of_education="bachelor's degree",
    #     lunch="standard",
    #     test_preparation_course="none",
    #     reading_score=72,
    #     writing_score=74
    # )

    # # Convert sample data to DataFrame
    # input_df = sample_data.get_data_as_df()
    # print("Input DataFrame:")
    # print(input_df)

    # # Initialize Predict_pipeline
    # pipeline = Predict_pipeline()

    # # Make prediction
    # prediction = pipeline.predict(input_df)
    # print("Prediction:")
    # print(prediction)
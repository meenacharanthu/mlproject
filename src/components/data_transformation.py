import sys
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self, num_cols, cat_cols):
        try:
            logging.info('Data Transformation Started')
            
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('std_scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_cols),
                ('cat', cat_pipeline, cat_cols)
            ])

            logging.info('Data Transformation Completed')
            return preprocessor

        except Exception as e:
            logging.error(f'Error in Data Transformation: {str(e)}')
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('train and test data loaded')

            target_col = 'math score'
            input_feature_train_df = train_df.drop(target_col, axis=1)
            target_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(target_col, axis=1)
            target_test_df = test_df[target_col]

            num_cols = input_feature_train_df.select_dtypes(exclude='object').columns
            cat_cols = input_feature_test_df.select_dtypes(include='object').columns

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            preprocessing_obj = self.get_data_transformer_obj(num_cols, cat_cols)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_df)]
            
            logging.info('Saving preprocessor object')
            
            save_obj(
                obj=preprocessing_obj, 
                file_path = self.transformation_config.preprocessor_path
            )
            return train_arr, test_arr, preprocessing_obj


        except Exception as e:
            raise CustomException(e,sys)

    




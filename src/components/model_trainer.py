import os
import sys
import pandas as pd
import numpy as np
import pickle
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from src.utils import save_obj, evaluate_models
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score




@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Model Training Started')
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                           models=models, param=params)


            best_model_score = max([report['test_score'] for report in model_report.values()])
            logging.info(f'Best model score: {best_model_score}')

            best_model_name = [name for name, report in model_report.items() if report['test_score'] == best_model_score][0]

            if best_model_score < 0.6:
                raise CustomException('No best model found')

            logging.info(f'Best model found: {best_model_name} with score: {best_model_score}')

            best_model = models[best_model_name]
            best_params = model_report[best_model_name]['best_params']
            best_model.set_params(**best_params)
            best_model.fit(X_train, y_train)
            predicted = best_model.predict(X_test)

            save_obj(
                obj=best_model,
                file_path=self.model_trainer_config.model_path
            )

            r2_square = r2_score(y_test, predicted)
            logging.info(f'R2 Score: {r2_square}')
            return r2_square

        except Exception as e:
            logging.error(f"Error in initiate_model_trainer method: {e}")
            raise CustomException(e, sys)
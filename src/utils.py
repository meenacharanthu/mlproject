import sys
import os
import pandas as pd
import numpy as np
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

def save_obj(obj, file_path):
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir, exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
    except Exception as e:
        CustomException(e,sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            model_name = list(models.keys())[i]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            logging.info(f'best parameters for {model_name}: {gs.best_params_}')
            logging.info(f'best score for {model_name}: {gs.best_score_}')

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
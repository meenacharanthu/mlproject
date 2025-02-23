import sys
import os
import pandas as pd
import numpy as np
import pickle

from src.exception import CustomException
from src.logger import logging

def save_obj(obj, file_path):
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir, exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
    except Exception as e:
        CustomException(e,sys)

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import logging

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            from src.pipeline.predict_pipeline import CustomData, Predict_pipeline

            # Get data from form
            gender = request.form.get('gender')
            race_ethnicity = request.form.get('race_ethnicity')
            parental_level_of_education = request.form.get('parental_level_of_education')
            lunch = request.form.get('lunch')
            test_preparation_course = request.form.get('test_preparation_course')
            reading_score = int(request.form.get('reading_score'))
            writing_score = int(request.form.get('writing_score'))

            # Create CustomData instance
            sample_data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            # Convert sample data to DataFrame
            input_df = sample_data.get_data_as_df()
            logging.debug("Input DataFrame:")
            logging.debug(input_df)

            # Initialize Predict_pipeline
            pipeline = Predict_pipeline()

            # Make prediction
            prediction = pipeline.predict(input_df)
            logging.debug(f"Prediction: {prediction}")
        

            return render_template('home.html', prediction=prediction[0])
        except Exception as e:
            logging.error(f"Error in predict route: {e}")
            return render_template('home.html', error=str(e))
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
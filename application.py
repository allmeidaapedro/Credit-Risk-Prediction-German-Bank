'''
This script aims to build a simple web application using Flask. The web application will interact with the ML model artifacts such that we can make predictions by giving input features values.
'''

'''
Importing the libraries.
'''

# Web app.
from flask import Flask, request, render_template

# Data manipulation.
import numpy as np
import pandas as pd

# File handling.
import os

# Predictions.
from src.pipeline.predict_pipeline import InputData, PredictPipeline


application = Flask(__name__)


app = application


# Route for the home page.

@app.route('/')
def index():
    '''
    Route handler for the home page.

    This function handles the GET request for the home page. It renders the 'index.html' template, which serves as the
    homepage for the customer credit risk prediction web application.

    :return: The rendered home page.
    :rtype: str
    '''
    return render_template('index.html')


# Route for prediction page.

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    '''
    Route handler for predicting credit risk.

    This function handles the POST request for predicting credit risk based on input data. If the request is a GET request,
    the function renders the 'home.html' template. If the request is a POST request, it collects input data from the form,
    processes it to make a prediction, and returns the prediction result.

    :return: The prediction result.
    :rtype: str
    '''

    if request.method == 'GET':
        return render_template('home.html')
    else:
        input_data = InputData(
            age=request.form.get('age'),
            sex=request.form.get('sex'),
            job=request.form.get('job'),
            housing=request.form.get('housing'),
            saving_accounts=request.form.get('saving_accounts'),
            checking_account=request.form.get('checking_account'),
            credit_amount=request.form.get('credit_amount'),
            duration=request.form.get('duration'),
            purpose=request.form.get('purpose')
        )

        input_df = input_data.get_input_data_df()
        print(input_df)
        print('\nBefore prediction.')

        predict_pipeline = PredictPipeline()
        print('\nMid prediction')
        prediction = predict_pipeline.predict(input_df)
        print('\nAfter prediction.')

        return prediction
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)


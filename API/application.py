# Import libraries
import sys
import os
import shutil
import time
import traceback
import json
import logging
import traceback

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
# We will use joblib for model persistence (it is better than pickle)
from sklearn.externals import joblib
from helpers import round_predicted_results, extract_right_eye_data, round_predicted_results_2, merge_right_eye_predictions
from logging.handlers import RotatingFileHandler
from time import strftime
from authentication.authenticate import requires_auth
from train.right_eye import train_right_eye_sphere_model, train_right_eye_cyl_axis_model

application = Flask(__name__)

# These will be populated at training time
config = None
logger = logging.getLogger('__name__')

# Load configuration file
print("Loading application configuration...")

with open("config.json") as json_data:
    config = json.load(json_data)

print("Application configuration loaded.")

# Setup path variables
right_eye_sphere_columns = "{0}/{1}/{2}".format(
    config["model_directory"], "right_eye", "right_eye_sphere_columns.pkl")
right_eye_sphere_model = "{0}/{1}/{2}".format(
    config["model_directory"], "right_eye", "right_eye_sphere_model.pkl")

right_eye_cyl_axis_columns = "{0}/{1}/{2}".format(
    config["model_directory"], "right_eye", "right_eye_cyl_axis_columns.pkl")
right_eye_cyl_axis_model = "{0}/{1}/{2}".format(
    config["model_directory"], "right_eye", "right_eye_cyl_axis_model.pkl")

# Setup the CORS
cors = CORS(application, resources={
            r"/api/v1/*": {"origins": config["cors_origins"]}})

# Default endpoint
@application.route('/', methods=['GET'])
def index():
    return Response("---SVL ML API is active---")

# Route used to train the model
@application.route('/api/v1/od/train', methods=['GET'])
@requires_auth
def train():
    # Train the model - sphere
    model_columns, reg = train_right_eye_sphere_model(config)
    model_columns_2, reg_2 = train_right_eye_cyl_axis_model(config)

    if model_columns and reg and model_columns_2 and reg_2:
        # Capture a list of columns that will be used for prediction and trained model
        joblib.dump(model_columns, right_eye_sphere_columns)
        joblib.dump(reg, right_eye_sphere_model)
        joblib.dump(model_columns_2, right_eye_cyl_axis_columns)
        joblib.dump(reg_2, right_eye_cyl_axis_model)
    else:
        return Response('Model training failed', status=500)

    return Response('Model created/trained successfully', status=201)


# This is the route used for predictions
# Input data - all features used during the training process
@application.route('/api/v1/od/predict', methods=['POST'])
@requires_auth
def predict():
    try:
        try:
            # Load the sphere model
            print("Loading right eye sphere model...")
            right_eye_sph_model = joblib.load(right_eye_sphere_model)
            right_eye_sph_columns = joblib.load(right_eye_sphere_columns)
            print("Right eye sphere model loaded.")

            # Load the cyl-axis model
            print("Loading right eye cyl-axis model...")
            right_eye_cl_ax_model = joblib.load(right_eye_cyl_axis_model)
            right_eye_cl_ax_columns = joblib.load(right_eye_cyl_axis_columns)
            print("Right eye cyl-axis model loaded.")

        except Exception as e:
            print(str(e))
            right_eye_sph_model = None
            right_eye_cl_ax_model = None

        if right_eye_sph_model and right_eye_cyl_axis_model:
            
            # Get request data
            json_ = request.json
            
            right_eye_data = extract_right_eye_data(json_)

            # Create pandas dataframe from the request data
            query = pd.get_dummies(pd.DataFrame(right_eye_data))

            # Filter created dataframe with columns necessary for our model
            # (we have stored these columns during the training process)
            # In case there are some missing columns in request data fill them out with 0 value
            query = query.reindex(columns=right_eye_sph_columns, fill_value=0)

            # Make the prediction for the right eye - sphere
            print("Prediction for right eye sphere started...")
            prediction = round_predicted_results(list(right_eye_sph_model.predict(query)))
            print("Prediction for right eye sphere done.")

            query = pd.get_dummies(pd.DataFrame(right_eye_data))
            query = query.reindex(columns=right_eye_cl_ax_columns, fill_value=0)

            # Make the prediction for the right eye - cyl-axis
            print("Prediction for right eye cyl-axis started...")
            prediction2 = round_predicted_results_2(list(right_eye_cl_ax_model.predict(query)))
            print("Prediction for right cyl-axis done.")
            
            predictions = merge_right_eye_predictions(prediction, prediction2, json_)

            # Return predicted result (apply proper rounding to the predicted result)
            return jsonify({"predictions": predictions})
        else:
            print('Model is not trained!')
            return Response('Model is not trained yet. You can train the model by using /train endpoint.', status=400)
    except Exception as e:
        print(str(e))
        return Response("Error occured", status=500)


# Endpoint used to clear/delete trained model from file system and from memory
@application.route('/api/v1/od/clear', methods=['GET'])
@requires_auth
def clear():
    try:
        print("Clearing right eye model from file system and memory...")
        shutil.rmtree('model/right_eye')
        os.makedirs("{0}/{1}".format(config["model_directory"], "right_eye"))
        print("Right eye model cleared.")

        return Response('Right eye model cleared', 200)
    except Exception as e:
        print(str(e))
        return Response('Could not remove and recreate the model directory', status=500)

@application.before_request
def before_request():
    if request.method == "POST":
        logger.error('%s %s %s %s %s \n',
                         request.remote_addr,
                         request.method,
                         request.scheme,
                         request.full_path,
                         request.json)

@application.after_request
def after_request(response):
    # This if avoids the duplication of registry in the log,
    # since that 500 is already logged via @application.errorhandler
    if response.status_code != 500:
        ts = strftime('[%Y-%b-%d %H:%M]')
        logger.error('%s %s %s %s %s %s \n %s \n',
            ts,
            request.remote_addr,
            request.method,
            request.scheme,
            request.full_path,
            response.status,
            response.data.decode("utf-8"))
    return response


@application.errorhandler(Exception)
def exceptions(e):
    ts = strftime('[%Y-%b-%d %H:%M]')
    tb = traceback.format_exc()
    logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s',
                 ts,
                 request.remote_addr,
                 request.method,
                 request.scheme,
                 request.full_path,
                 tb)
    return "Internal Server Error", 500


# Script configuration
if __name__ == '__main__':
    try:
        handler = RotatingFileHandler(
            config["log_file_name"], maxBytes=config["log_file_max_size_in_bytes"], backupCount=5)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    except Exception as e:
        print(str(e))

    application.run()

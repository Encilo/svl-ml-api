# Import libraries
import sys
import os
import shutil
import time
import traceback
import json
import io
import logging
import traceback

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
from sklearn.externals import joblib # We will use joblib for model persistence (it is better than pickle)
from helpers import round_predicted_results
from boto3 import client
from logging.handlers import RotatingFileHandler
from time import strftime
from authentication.authenticate import requires_auth

application = Flask(__name__)

# These will be populated at training time
model_columns = None
reg = None
config = None
logger = logging.getLogger('__name__')

# Load configuration file
print("Loading application configuration...")

with open("config.json") as json_data:
    config = json.load(json_data)

print("Application configuration loaded.")

# Setup the CORS
cors = CORS(application, resources={r"/api/v1/*": {"origins": config["cors_origins"]}})

# Configure store location for model and columns definition
model_file_name = "{0}/{1}".format(config["model_directory"], config["model_name"]) 
model_columns_file_name = "{0}/{1}".format(config["model_directory"], config["model_columns_name"])

# Default endpoint
@application.route('/', methods=['GET'])
def index():
    return Response("---SVL ML API is active---") 

# Route used to train the model
@application.route('/api/v1/train', methods=['GET'])
@requires_auth
def train():

    # Use this two lines of code only if the s3 file is private
    # Configure s3 client
    print("Configuring S3 client...")

    s3 = client('s3', 
                aws_access_key_id = config["aws_access_key_id"], 
                aws_secret_access_key = config["aws_secret_access_key"]
               )

    print("S3 client configured.")

    # Get the data from the bucket
    print("Retrieving training data from S3 bucket - {0}/{1}".format(config["s3_bucket_name"], config["s3_file_name"]))
    
    bucket_file = s3.get_object(Bucket = config["s3_bucket_name"], Key = config["s3_file_name"])

    print("Training data retrieved.")
    print("Model training started...")

    # Import the dataset - 84 records
    # dataset = pd.read_csv(config["training_data_location"]) # use this line of code instead of the two lines above in case s file is publicly available
    dataset = pd.read_csv(io.BytesIO(bucket_file['Body'].read()))

    # Extract data for the right eye - sphere           
    columns = config["data_columns"]

    right_eye_sphere_dataset = pd.DataFrame(dataset, columns = columns)

    # check for duplicates and remove if exists
    duplicates_exists = right_eye_sphere_dataset.duplicated().any()
    if duplicates_exists:
        right_eye_sphere_dataset = right_eye_sphere_dataset.drop_duplicates() 

    # Map categorical data
    notes_map = {"happy": 1, "unhappy": 0}
    right_eye_sphere_dataset["notes"] = right_eye_sphere_dataset["notes"].map(notes_map)

    # Create feature matrix
    X = right_eye_sphere_dataset.iloc[:,:-3]    

    # Create predicted vector
    y = right_eye_sphere_dataset.iloc[:,5].values

    # Split dataset to train and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # Multiple Linear Regression - Train the model
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(X.columns)
    joblib.dump(model_columns, model_columns_file_name)

    # capture trained model
    global reg
    reg = regressor
    start = time.time()

    print('Trained in %.1f seconds' % (time.time() - start))
    print("Model training done.")

    joblib.dump(reg, model_file_name)

    return Response('Model created/trained successfully', status = 201)


# This is the route used for predictions
# Input data - all features used during the training process
@application.route('/api/v1/predict', methods=['POST'])
@requires_auth
def predict():
    # Check if the model is trained
    if reg:
        try:
            # Get request data
            json_ = request.json

            # Create pandas dataframe from the request data
            query = pd.get_dummies(pd.DataFrame(json_))

            # Filter created dataframe with columns necessary for our model 
            # (we have stored these columns during the training process)
            # In case there are some missing columns in request data fill them out with 0 value
            query = query.reindex(columns = model_columns, fill_value = 0)

            # Make the prediction
            print("Prediction started...")
            prediction = list(reg.predict(query))
            print("Prediction done.")

            # Return predicted result (apply proper rounding to the predicted result)
            return jsonify({ "predictions" : round_predicted_results(prediction, json_)})

        except Exception as e:
            print(str(e))
            return Response("Error occured", status = 500)
    else:
        print('Model is not trained!')
        return Response('Model is not trained yet. You can train the model by using /train endpoint.', status = 400)

# Endpoint used to clear/delete trained model from file system and from memory
@application.route('/api/v1/clear', methods=['GET'])
@requires_auth
def wipe():
    try:
        print("Clearing model from file system and memory...")
        shutil.rmtree('model')
        os.makedirs(config["model_directory"])
        global model_columns
        global reg
        model_columns = None        
        reg = None
        print("Model cleared.")
        return Response('Model cleared', 200)
    except Exception as e:
        print(str(e))
        return Response('Could not remove and recreate the model directory', status = 500)

@application.after_request
def after_request(response):
    # This if avoids the duplication of registry in the log,
    # since that 500 is already logged via @application.errorhandler
    if response.status_code != 500 :
        ts = strftime('[%Y-%b-%d %H:%M]')
        if request.method == "POST" :
            logger.error('%s %s %s %s %s %s \n %s \n %s \n',
                    ts,
                    request.remote_addr,
                    request.method,
                    request.scheme,
                    request.full_path,
                    response.status,
                    request.json,
                    response.data.decode("utf-8"))
        else:
            logger.error('%s %s %s %s %s %s \n %s \n %s \n',
                    ts,
                    request.remote_addr,
                    request.method,
                    request.scheme,
                    request.full_path,
                    response.status,
                    request,
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
        handler = RotatingFileHandler(config["log_file_name"], maxBytes = config["log_file_max_size_in_bytes"], backupCount = 5) 
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)  
        port = int(sys.argv[1])

    except Exception as e:
        port = config["default_port"]

    try:
        reg = joblib.load(model_file_name)
        print('Model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('Model columns loaded')

    except Exception as e:
        print('Model is not trained')
        print(str(e))
        reg = None

    application.run()

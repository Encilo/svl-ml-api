import io
import pandas as pd

from boto3 import client


def train_right_eye_sphere_model(config):

    try:
        print("Model training started...")

        # Import the dataset
        bucket_file = get_training_data(config)
        dataset = pd.read_csv(io.BytesIO(bucket_file['Body'].read()))

        # Extract data for the right eye - sphere
        columns = config["data_set_columns"]["right_eye_sphere"]

        right_eye_sphere_dataset = pd.DataFrame(dataset, columns=columns)

        # check for duplicates and remove if exists
        duplicates_exists = right_eye_sphere_dataset.duplicated().any()
        if duplicates_exists:
            right_eye_sphere_dataset = right_eye_sphere_dataset.drop_duplicates()

        # Map categorical data
        notes_map = {"happy": 1, "unhappy": 0}
        right_eye_sphere_dataset["notes"] = right_eye_sphere_dataset["notes"].map(
            notes_map)

        # Create feature matrix
        X = right_eye_sphere_dataset.iloc[:, :-3]

        # Create predicted vector
        y = right_eye_sphere_dataset.iloc[:, 5].values

        # Split dataset to train and test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # Multiple Linear Regression - Train the model
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        print("Model training done.")

        return list(X.columns), regressor
    except Exception as e:
        print(str(e))
        return None, None
    


def train_right_eye_cyl_axis_model(config):
    try:
        print("Model training started...")

        # Import the dataset
        bucket_file = get_training_data(config)
        dataset = pd.read_csv(io.BytesIO(bucket_file['Body'].read()))

        # Extract data for the right eye - cyl/axis
        columns = config["data_set_columns"]["right_eye_cyl_axis"]

        right_eye_dataset = pd.DataFrame(dataset, columns=columns)

        # Check for duplicates and remove if exists
        duplicates_exists = right_eye_dataset.duplicated().any()
        if duplicates_exists:
            right_eye_dataset = right_eye_dataset.drop_duplicates()

        # map categorical data
        notes_map = {"happy": 1, "unhappy": 0}
        right_eye_dataset["notes"] = right_eye_dataset["notes"].map(notes_map)

        # Create feature matrix
        X = right_eye_dataset.iloc[:, :-3]

        # Create predicted matrix
        y = right_eye_dataset.iloc[:, 7:9]

        # Split dataset to train and test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # SVR - Train the model
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
        regressor = MultiOutputRegressor(SVR(kernel = "linear"), n_jobs = -1)
        regressor.fit(X_train, y_train)

        print("Model training done.")

        return list(X.columns), regressor
    except Exception as e:
        print(str(e))
        return None, None


def get_training_data(config):
    try:
        # Use this two lines of code only if the s3 file is private
        # Configure s3 client
        print("Configuring S3 client...")

        s3 = client('s3',
                    aws_access_key_id=config["aws_access_key_id"],
                    aws_secret_access_key=config["aws_secret_access_key"]
                    )

        print("S3 client configured.")

        # Get the data from the bucket
        print("Retrieving training data from S3 bucket - {0}/{1}".format(
            config["s3_bucket_name"], config["s3_file_name"]))

        bucket_file = s3.get_object(
            Bucket=config["s3_bucket_name"], Key=config["s3_file_name"])

        print("Training data retrieved.")

        return bucket_file
    except Exception as e:
        print(str(e))

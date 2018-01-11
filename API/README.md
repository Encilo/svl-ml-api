# Flask API for SVL ML models
A simple Flask API that can serve predictions using ML model. 
Reads a pickled model into memory when the Flask app is started and returns predictions through the /predict endpoint. 
You can also use the /train endpoint to train/retrain the model and the /clear endpoint to remove trained model from the file disk and the memory. 

### Dependencies
boto3==1.5.10
botocore==1.8.24
click==6.7
docutils==0.14
Flask==0.12.2
Flask-Cors==3.0.3
itsdangerous==0.24
Jinja2==2.10
jmespath==0.9.3
MarkupSafe==1.0
numpy==1.14.0
pandas==0.22.0
python-dateutil==2.6.1
pytz==2017.3
s3transfer==0.1.12
scikit-learn==0.19.1
scipy==1.0.0
six==1.11.0
Werkzeug==0.14.1

```
pip install -r requirements.txt
```

### Running API
```
python application.py
```

# Endpoints
### /predict (POST)
Returns an JSON object representing predictions. Here's a sample input:
```
[
    {
    	"visit_id": "123",
        "age": 23,
        "notes": "unhappy",
        "visualAcuity_1": 0.8,
        "lensometer_1": -1.25,
        "autorefraction_1": -2.25
    },
    {
    	"visit_id": "456",
        "age": 42,
        "notes": "happy",
        "visualAcuity_1": 0.8,
        "lensometer_1": -1.00,
        "autorefraction_1": -0.25
    }
]
```

and sample output:
```
{
    "predictions": [
        {
            "prediction": -1.5,
            "visit_id": "123"
        },
        {
            "prediction": -0.75,
            "visit_id": "456"
        }
    ]
}
```

### /train (GET)
Trains the model. This is currently hard-coded to be a LinearRegression model that is run on a ModernDay dataset.

### /clear (GET)
Removes the trained model and the model should be retrained before /predict endpoint is used.

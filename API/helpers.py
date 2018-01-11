import jwt
import json

config = None

# Load configuration file
with open("config.json") as json_data:
    config = json.load(json_data)

# Method used to round  predicted results (min step 0.25)
def round_predicted_results(predicted, request_data):
    index = 0
    for x in predicted:
        tmp = int((x - int(x)) * 100)
        if tmp < 0:
            tmp = tmp * (-1)
        v1 = int(tmp / 25)
        v2 = tmp % 25
        if v2 > 12:
            v1 = v1 + 1
        if x < 0:
            # predicted[index] = int(x) - (v1 * 0.25)
            predicted[index] = { "prediction": int(x) - (v1 * 0.25), "visit_id": request_data[index]["visit_id"] }
        else:
            # predicted[index] = int(x) + (v1 * 0.25)
            predicted[index] = { "prediction": int(x) + (v1 * 0.25), "visit_id": request_data[index]["visit_id"] }
        index = index + 1
    return predicted

# Method used to verify JWT token
def verify_jwt_token(token):

    audience = config["jwt_validation"]["full_validation"]["audience"]
    issuer = config["jwt_validation"]["full_validation"]["issuer"]
    jwt_signature_secret = config["jwt_validation"]["jwt_signature_secret"]
    algorithm = config["jwt_validation"]["jwt_algorithm"]

    try:
        if audience and issuer:
            decoded_token = jwt.decode(token, jwt_signature_secret, audience = audience, issuer = issuer, algorithm = algorithm)
        else:
            decoded_token = jwt.decode(token, jwt_signature_secret, algorithm = algorithm, options = {'verify_aud': False})
        return True
    except Exception as e:
        print(str(e))
        return False
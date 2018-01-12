import jwt
import json

config = None

# Load configuration file
with open("config.json") as json_data:
    config = json.load(json_data)

# Method used to round  predicted results (min step 0.25)
def round_predicted_results(predicted):
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
            predicted[index] = int(x) - (v1 * 0.25)
        else:
            predicted[index] = int(x) + (v1 * 0.25)
        index = index + 1
    return predicted

# Method used to round  predicted results (min step 0.25)
def round_predicted_results_2(predicted):
    result = []
    for x in predicted:
        tmp = int((x[0] - int(x[0])) * 100)
        tmp2 = x[1] - int(x[1])

        if tmp < 0:
            tmp = tmp * (-1)

        if tmp2 < 0:
            tmp2 = tmp2 * (-1)

        if tmp2 <= 0.50:
            x[1] = int(x[1])
        else:
            x[1] = int(x[1]) + 1

        v1 = int(tmp / 25)
        v2 = tmp % 25
        if v2 > 12:
            v1 = v1 + 1
        if x[0] < 0:
            result.append([int(x[0]) - (v1 * 0.25), x[1]])
        else:
            result.append([int(x[0]) + (v1 * 0.25), x[1]])

    return result

# Method used to verify JWT token
def verify_jwt_token(token):

    audience = config["jwt_validation"]["full_validation"]["audience"]
    issuer = config["jwt_validation"]["full_validation"]["issuer"]
    jwt_signature_secret = config["jwt_validation"]["jwt_signature_secret"]
    algorithm = config["jwt_validation"]["jwt_algorithm"]

    try:
        if audience and issuer:
            decoded_token = jwt.decode(
                token, jwt_signature_secret, audience=audience, issuer=issuer, algorithm=algorithm)
        else:
            decoded_token = jwt.decode(
                token, jwt_signature_secret, algorithm=algorithm, options={'verify_aud': False})
        return True
    except Exception as e:
        print(str(e))
        return False

# Method used to extract right eye data from the request
def extract_right_eye_data(request_data):
    model_data = []

    for visit in request_data:
        model_data.append(visit["od"])

    return model_data

# Method used to merge right eye predictions sphere with cyl/axis
def merge_right_eye_predictions(prediction, prediction_2, request_data):
    result = []
    index = 0
    for x in request_data:
        result.append({"visit_id": request_data[index]["visit_id"], "od": {
                      "sphere": prediction[index], "cyl": prediction_2[index][0], "axis": prediction_2[index][1]}})
        index = index + 1
    return result

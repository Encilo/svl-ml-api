import json

from functools import wraps
from flask import request, Response
from helpers import verify_jwt_token

config = None

# Load configuration file
with open("config.json") as json_data:
    config = json.load(json_data)

def check_auth(username, password):
    # This function is called to check username and password
    return username == config["authentication_username"] and password == config["authentication_password"]

def unauthorized():
    # Sends a 401 response that enables basic auth
    return Response("Unauthorized", 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})

# Method used to secure protected endpoints
#def requires_auth(f):
#    @wraps(f)
#    def decorated(*args, **kwargs):
#        auth = request.authorization
#        if not auth or not check_auth(auth.username, auth.password):
#            return unauthorized()
#        return f(*args, **kwargs)
#    return decorated

# Method used to secure protected endpoints
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if auth:
            if not check_auth(auth.username, auth.password):
                return unauthorized()
        else:
            auth = request.headers["Authorization"]
            if auth:
                token = auth.replace("Bearer ","")
                if not verify_jwt_token(token):
                    return unauthorized()
            else:
                return unauthorized()
        return f(*args, **kwargs)
    return decorated
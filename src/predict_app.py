"""House price prediction service"""

from flask import Flask, request
import numpy as np
from flask_httpauth import HTTPTokenAuth
from flask_cors import CORS
from joblib import load
from dotenv import dotenv_values

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'models/Catboost_top.joblib'
config = dotenv_values(".env")
auth = HTTPTokenAuth(scheme='Bearer')

tokens = {
    config["APP_TOKEN"]: "radmuire",
}

@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]

def predict(in_data: dict) -> int:
    """ Predict house price from input data parameters.
    :param in_data: house parameters.
    :raise Error: If something goes wrong.
    :return: House price, RUB.
    :rtype: int
    """
    ['rooms_count', 'author_type', 'floor', 'floors_count', 'first_floor', 'last_floor', 'total_meters', 'district']
    rooms_count = int(in_data['rooms_count'])
    author_type = str(in_data['author_type'])
    floor = int(in_data['floor'])
    floors_count = int(in_data['floors_count'])
    first_floor = (floor == 1)*1
    last_floor = (floor == floors_count)*1
    total_meters = float(in_data['total_meters'])
    district = int(in_data['district'])

    input_features = [
        rooms_count,
        author_type,
        floor,
        floors_count,
        first_floor,
        last_floor,
        total_meters,
        district
    ]
    model = load(MODEL_PATH)
    res = np.exp(model.predict([input_features]))
    return int(res)


@app.route("/")
def home():
    return '<h1>Housing price service.</h1> Use /predict endpoint'


@app.route("/predict", methods=['POST'])
@auth.login_required
def predict_web_serve():
    """Dummy service"""
    in_data = request.get_json()
    price = predict(in_data)
    return {'price': price}


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
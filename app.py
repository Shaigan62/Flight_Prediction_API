from flask import Flask, jsonify, request

import Flight_Predict

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict_flight():
    flight_info = request.get_json()
    prediction = Flight_Predict.predict_data(flight_info)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run()

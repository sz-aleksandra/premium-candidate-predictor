import pickle
from flask import Flask, request, jsonify
import logging
import json


BASE_MODEL_PATH = 'content/results/logistic_regression.pkl'
ADVANCED_MODEL_PATH = 'content/results/random_forest.pkl'
SCALER_PATH = 'content/results/data_scaler.pkl'
ATTRIBUTES_NEEDED_INFO_PATH = 'content/custom_data/attributes_required.json'
HOSTING_IP = '0.0.0.0'
HOSTING_PORT = 8080




app = Flask(__name__)

logging.basicConfig(
    filename='content/results/ab_test.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    with open(SCALER_PATH,'rb') as f:
        scaler = pickle.load(f)
    with open(BASE_MODEL_PATH, 'rb') as f:
        base_model = pickle.load(f)
    with open(ADVANCED_MODEL_PATH, 'rb') as f:
        advanced_model = pickle.load(f)
except FileNotFoundError as e:
    logging.error(f"Nie udało się załadować modeli: {e}")
    raise RuntimeError("Nie udało się załadować modeli. Sprawdź pliki .pkl.")

with open(ATTRIBUTES_NEEDED_INFO_PATH, 'r') as file:
    NUMERIC_FEATURES = json.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Brak danych wejściowych."}), 400

        try:
            features = [data[feature] for feature in NUMERIC_FEATURES]
        except KeyError as e:
            missing_feature = e.args[0]
            return jsonify({"error": f"Brakuje wymaganej cechy: {missing_feature}"}), 400
        features_scaled = scaler.transform([features])
        base_prediction = base_model.predict(features_scaled)[0]
        advanced_prediction = advanced_model.predict(features_scaled)[0]

        logging.info(
            f"Zapytanie: {data}, Base Prediction: {base_prediction}, Advanced Prediction: {advanced_prediction}"
        )

        return jsonify({
            "predictions": {
                "base": float(base_prediction),
                "advanced": float(advanced_prediction)
            }
        })

    except Exception as e:
        logging.error(f"Wystąpił błąd: {str(e)}")
        return jsonify({"error": "Wewnętrzny błąd serwera."}), 500
    

@app.route('/predict_base', methods=['POST'])
def predict_base():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Brak danych wejściowych."}), 400

        try:
            features = [data[feature] for feature in NUMERIC_FEATURES]
        except KeyError as e:
            missing_feature = e.args[0]
            return jsonify({"error": f"Brakuje wymaganej cechy: {missing_feature}"}), 400
        features_scaled = scaler.transform([features])
        base_prediction = base_model.predict(features_scaled)[0]

        logging.info(
            f"Zapytanie: {data}, Base Prediction: {base_prediction}"
        )

        return jsonify({
            "base_prediction": float(base_prediction)
        })

    except Exception as e:
        logging.error(f"Wystąpił błąd: {str(e)}")
        return jsonify({"error": "Wewnętrzny błąd serwera."}), 500
    

@app.route('/predict_advanced', methods=['POST'])
def predict_advanced():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Brak danych wejściowych."}), 400

        try:
            features = [data[feature] for feature in NUMERIC_FEATURES]
        except KeyError as e:
            missing_feature = e.args[0]
            return jsonify({"error": f"Brakuje wymaganej cechy: {missing_feature}"}), 400
        features_scaled = scaler.transform([features])
        advanced_prediction = advanced_model.predict(features_scaled)[0]

        logging.info(
            f"Zapytanie: {data}, Advanced Prediction: {advanced_prediction}"
        )

        return jsonify({
            "advanced_prediction": float(advanced_prediction)
        })

    except Exception as e:
        logging.error(f"Wystąpił błąd: {str(e)}")
        return jsonify({"error": "Wewnętrzny błąd serwera."}), 500

if __name__ == '__main__':
    app.run(host=HOSTING_IP, port=HOSTING_PORT, debug=True)



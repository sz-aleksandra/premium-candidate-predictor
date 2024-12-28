import pickle
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

logging.basicConfig(
    filename='ab_test.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    with open('data_scaler.pkl','rb') as f:
        scaler = pickle.load(f)
    with open('lr_model.pkl', 'rb') as f:
        base_model = pickle.load(f)

    with open('rfc_model.pkl', 'rb') as f:
        advanced_model = pickle.load(f)
except FileNotFoundError as e:
    logging.error(f"Nie udało się załadować modeli: {e}")
    raise RuntimeError("Nie udało się załadować modeli. Sprawdź pliki .pkl.")

NUMERIC_FEATURES = [
    "frequency", "avr_len", "likes", "skips", "plays", "ads", "ad_quits",
    "artist_diversity", "artist_gini", "year", "popularity", "explicit",
    "danceability", "energy", "key", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature",
    "favourite_genres_count"
]

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)



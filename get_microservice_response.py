import pandas as pd
import requests
from sklearn.metrics import accuracy_score

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, encoding_errors='ignore')
    return data

def get_base_prediction_from_microservice(X):
    response = requests.post("http://localhost:8080/predict_base", json=X)
    return response.json()['base_prediction']

def get_advanced_prediction_from_microservice(X):
    response = requests.post("http://localhost:8080/predict_advanced", json=X)
    return response.json()['advanced_prediction']

def main():
    NUMBER_OF_SAMPLES_TO_TEST = 10
    X_test = load_and_preprocess_data('content/custom_data/X_test.csv')
    print(f"Number of all available test samples: {len(X_test)}")
    Y_test = load_and_preprocess_data('content/custom_data/Y_test.csv').to_dict(orient='records')
    X_test = X_test.head(NUMBER_OF_SAMPLES_TO_TEST).to_dict(orient='records')
    
    Y_base = []
    Y_adv = []
    Y_data = []
    for i, entry in enumerate(X_test):
        base_pred = get_base_prediction_from_microservice(entry)
        adv_pred = get_advanced_prediction_from_microservice(entry)
        y = Y_test[i]["premium_user"]
        Y_base.append(base_pred)
        Y_adv.append(adv_pred)
        Y_data.append(y)
        print(f"Sample {i}: Base={base_pred}, Advanced={adv_pred}, Data={y}")
    print(f"\nACCURACY:")
    print(f"Base model: {accuracy_score(Y_data, Y_base)}")
    print(f"Advanced model: {accuracy_score(Y_data, Y_adv)}")


if __name__ == "__main__":
    main()
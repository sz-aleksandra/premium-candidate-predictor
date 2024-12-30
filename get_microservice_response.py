
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score
from scipy import stats
import json
from numpy import var


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, encoding_errors='ignore')
    return data

def get_base_prediction_from_microservice(X):
    response = requests.post("http://localhost:8080/predict_base", json=X)
    return response.json()['base_prediction']



def main():
    X_train = load_and_preprocess_data('content/custom_data/X_train.csv')
    X_train = X_train.to_dict(orient='records')
    X_val = load_and_preprocess_data('content/custom_data/X_val.csv')
    X_val = X_val.to_dict(orient='records')
    Y_train = load_and_preprocess_data('content/custom_data/Y_train.csv')
    Y_val = load_and_preprocess_data('content/custom_data/Y_val.csv')

    


if __name__ == "__main__":
    main()
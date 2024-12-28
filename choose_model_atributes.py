import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, encoding_errors='ignore')
    return data

def show_attribute_weights(X, model, feature_names=None):
    if feature_names is None:
        feature_names = [f'{str(i)}' for i in X.columns]
    
    weights = model.coef_[0]
    feat_weights = list(zip(feature_names, weights))
    feat_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    sorted_features, sorted_weights = zip(*feat_weights)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(sorted_weights)), sorted_weights, color='skyblue')
    
    for i, weight in enumerate(sorted_weights):
        if weight < 0:
            bars[i].set_color('lightcoral')
    
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Linear Model Coefficients (Sorted by Magnitude)')
    plt.xticks(range(len(sorted_weights)), sorted_features, rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def train_and_evaluate_model(X_train, X_test, Y_train, Y_test, model, model_name):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")
    print(f"ROC AUC: {roc_auc_score(Y_test, Y_pred)}")
    print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
    
    with open(f'{model_name.lower().replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model, scaler

def main():
    X = load_and_preprocess_data('content/custom_data/processed_X.csv')
    Y = load_and_preprocess_data('content/custom_data/processed_Y.csv')
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    lr_model = LogisticRegression(max_iter=1000)
    lr_model, scaler = train_and_evaluate_model(
        X_train, X_test, Y_train, Y_test, 
        lr_model, "Logistic Regression"
    )
    
    show_attribute_weights(X, lr_model)

if __name__ == "__main__":
    main()
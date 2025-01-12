import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, encoding_errors='ignore')
    return data

def train_and_evaluate_model(X_train,X_val,Y_train,Y_val, scaler,model, model_name, save_model_path ="content/results/"):
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model.fit(X_train_scaled, Y_train)
    Y_pred_val = model.predict(X_val_scaled)
    Y_pred_train = model.predict(X_train_scaled)

    
    print(f"\n{model_name} Results:")
    print("Training data")
    print(f"Accuracy: {accuracy_score(Y_train, Y_pred_train)}")
    print(f"Classification Report:\n{classification_report(Y_train, Y_pred_train)}")

    print("Validation data")
    print(f"Accuracy: {accuracy_score(Y_val, Y_pred_val)}")
    print(f"Classification Report:\n{classification_report(Y_val, Y_pred_val)}")
    with open(f'{save_model_path}{model_name.lower().replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

def create_scaler(X_train,Y_train, scaler_name,save_model_path ="content/results/"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    with open(f'{save_model_path}{scaler_name.lower().replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return scaler

def main():
    X_train = load_and_preprocess_data('content/custom_data/X_train.csv')
    X_val = load_and_preprocess_data('content/custom_data/X_val.csv')
    Y_train = load_and_preprocess_data('content/custom_data/Y_train.csv')
    Y_val = load_and_preprocess_data('content/custom_data/Y_val.csv')

    scaler = create_scaler(X_train.values,Y_train.values, "data_scaler")

    lr_model = LogisticRegression()
    lr_model = train_and_evaluate_model(X_train, X_val, Y_train, Y_val, scaler,lr_model, "Logistic Regression")
    
    rf_model = RandomForestClassifier(criterion="entropy",n_estimators=100, random_state=42)
    rf_model = train_and_evaluate_model(X_train, X_val, Y_train, Y_val,scaler,rf_model , "Random Forest")

if __name__ == "__main__":
    main()
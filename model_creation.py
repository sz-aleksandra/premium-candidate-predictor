import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset."""
    data = pd.read_csv(filepath, encoding_errors='ignore')
    
    le = LabelEncoder()
    data['premium_user'] = le.fit_transform(data['premium_user'])
    return data

def train_and_evaluate_model(X_train,X_test,Y_train,Y_test, scaler,model, model_name):
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")
    print(f"ROC AUC: {roc_auc_score(Y_test, Y_pred)}")
    print(f"Classification Report:\n{classification_report(Y_test, Y_pred)}")
    
    with open(f'{model_name.lower().replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

def create_scaler(X_train,Y_train, scaler_name):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    with open(f'{scaler_name.lower().replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return scaler

def main():
    data = load_and_preprocess_data('content/processed_data.csv')
    Y = data['premium_user']
    X = data.drop('premium_user', axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    scaler = create_scaler(X_train,Y_train, "data_scaler")

    lr_model = LogisticRegression()
    lr_model = train_and_evaluate_model(X_train, X_test, Y_train, Y_test, scaler,lr_model, "Logistic Regression")
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model = train_and_evaluate_model(X_train, X_test, Y_train, Y_test,scaler,rf_model , "Random Forest")

if __name__ == "__main__":
    main()
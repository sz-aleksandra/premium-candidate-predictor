import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pickle

# Model 1: Logistic Regression
data = pd.read_csv('content/processed_data.csv', encoding_errors='ignore')

label_encoder = LabelEncoder()
data['premium_user'] = label_encoder.fit_transform(data['premium_user'])

data['favourite_genres_count'] = data['favourite_genres'].apply(lambda x: len(eval(x)))
data.drop('favourite_genres', axis=1, inplace=True)
data.drop(['user_id', 'city'], axis=1, inplace=True)

X = data.drop('premium_user', axis=1)
y = data['premium_user']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

with open('lr_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Model 2: Random Forest Classifier

label_encoder = LabelEncoder()
data['premium_user'] = label_encoder.fit_transform(data['premium_user'])

X = np.reshape(data['frequency'].values, (-1, 1))
y = data['premium_user']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)

print("Random Forest Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

with open('rfc_model.pkl', 'wb') as f:
    pickle.dump(model, f)

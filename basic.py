import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

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

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
import joblib
import datetime
import numpy as np

df = pd.read_csv('./cleaned_bank_data.csv')

X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

y_proba = model.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

f1_scores = 2 * (precision * recall) / np.maximum((precision + recall), 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

y_pred = (y_proba >= best_threshold).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

coef = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': model.coef_[0]
}).sort_values(by='Coeficiente', ascending=False)

print(coef.head(10))

# joblib.dump(model, f'logistic_model_.pkl')
joblib.dump(model, f"logistic_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")

joblib.dump(scaler, f'scaler_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')

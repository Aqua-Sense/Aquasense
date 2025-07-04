import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===================== Load the dataset =====================

dataset_path = "/kaggle/input/d/kerollosehab/final-gp-df/water_potability.csv"
data = pd.read_csv(dataset_path)

# ===================== Handle missing values =====================

data.fillna(data.median(), inplace=True)

# ===================== Separate features and target =====================

X = data.drop(columns='Potability')
y = data['Potability']

# ===================== Class distribution before split =====================

print("Class distribution before split:")
print(y.value_counts())

# ===================== Split the dataset =====================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nClass distribution in training set:")
print(y_train.value_counts())

# ===================== Train the model =====================

RF_model = RandomForestClassifier(class_weight='balanced')
RF_model.fit(X_train, y_train)

# ===================== Prediction and evaluation =====================

y_pred = RF_model.predict(X_test)

print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ===================== Plot feature importance =====================

importances = RF_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='teal')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score
)
from xgboost import XGBClassifier

# 1. Load dataset
df = pd.read_csv("/kaggle/input/d/kerollosehab/final-gp-df/Watera.csv")
print(f"✅ Dataset loaded with shape: {df.shape}")

# 2. Define features and target
features = ['ph', 'tds', 'conductivity', 'turbidity', 'salinity']
target = 'potability'

# 3. Calculate salinity based on conductivity
def calculate_salinity(conductivity_uS_cm):
    C_mS_cm = conductivity_uS_cm / 1000.0
    if C_mS_cm < 2.0:
        return conductivity_uS_cm * 0.0005
    return (
        0.008
        - 0.1692 * C_mS_cm
        + 25.3851 * C_mS_cm**2
        + 14.0941 * C_mS_cm**3
        - 7.0261 * C_mS_cm**4
        + 2.7081 * C_mS_cm**5
    )

df['salinity'] = df['conductivity'].apply(calculate_salinity)

# 4. Clean data
df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=features + [target])

# 5. Downsampling to balance the dataset
class_0 = df[df[target] == 0]
class_1 = df[df[target] == 1]
downsampled_class_0 = class_0.sample(n=len(class_1), random_state=42)
df_balanced = pd.concat([downsampled_class_0, class_1]).sample(frac=1, random_state=42)

print("\n📉 After Downsampling:")
print(df_balanced[target].value_counts())

# 6. Split data
X = df_balanced[features].values
y = df_balanced[target].astype(int).values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train XGBoost model
print("\n🚀 Training XGBoost model...")
xgb_model = XGBClassifier(
    scale_pos_weight=1.0,  # Adjust if needed
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# 9. Evaluate XGBoost model
y_pred = xgb_model.predict(X_test_scaled)
y_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("\n📋 XGBoost Classification Report:")
print(classification_report(y_test, y_pred, digits=3))
print(f"XGBoost ROC AUC Score: {roc_auc_score(y_test, y_probs):.3f}")

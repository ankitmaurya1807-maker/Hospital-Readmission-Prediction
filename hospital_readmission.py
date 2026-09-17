import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# 1. Load dataset
df = pd.read_csv("diabetic_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# 2. Create target
# <30 = readmitted within 30 days
df["target"] = df["readmitted"].apply(
    lambda x: 1 if x == "<30" else 0
)

print("\nTarget distribution:")
print(df["target"].value_counts())


# 3. Select features
features = [
    "age",
    "time_in_hospital",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "diag_1",
    "diag_2",
    "diag_3"
]

X = df[features]
y = df["target"]


# 4. Numerical columns
numeric_columns = [
    "time_in_hospital",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]


# 5. Categorical columns
categorical_columns = [
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "diag_1",
    "diag_2",
    "diag_3"
]


# 6. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)


# 7. Logistic Regression with L2 regularization
model = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=1000
)


# 8. Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 9. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# 10. Train model
pipeline.fit(X_train, y_train)


# 11. Predictions
y_pred = pipeline.predict(X_test)

y_prob = pipeline.predict_proba(X_test)[:, 1]


# 12. ROC-AUC
auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC:", auc)


# 13. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 14. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
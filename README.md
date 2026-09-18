# Hospital Readmission Prediction Using Logistic Regression

## 1. Project Overview

Hospital readmission prediction is a machine learning classification problem where the goal is to predict whether a patient will be **readmitted to the hospital within 30 days**.

This project uses **Logistic Regression with L2 regularization** to predict 30-day hospital readmission from patient encounter data.

The project is based on the diabetes hospital encounter dataset and focuses on identifying patterns associated with 30-day readmission.

---

## 2. Problem Statement

Build a machine learning model that predicts whether a patient will be readmitted to the hospital within 30 days.

The target variable is converted into:

```text
0 → Not readmitted within 30 days
1 → Readmitted within 30 days
```

The model is evaluated using **ROC-AUC, confusion matrix, precision, recall, and F1-score**.

---

## 3. Objectives

* Load and explore the hospital patient dataset.
* Create a binary target for 30-day readmission.
* Select relevant patient and hospital-encounter features.
* Separate numerical and categorical features.
* Scale numerical features.
* Convert categorical features into numerical form using One-Hot Encoding.
* Split the data into training and testing sets.
* Train a Logistic Regression model.
* Apply **L2 regularization**.
* Predict 30-day readmission.
* Evaluate the model using:

  * ROC-AUC
  * Confusion Matrix
  * Precision
  * Recall
  * F1-score

---

## 4. Dataset

The project uses the **Diabetes 130-US Hospitals** dataset, which contains hospital encounter information for diabetic patients.

The original research dataset was obtained from the Cerner Health Facts database and contains patient encounter information and multiple clinical and administrative features.

The dataset includes information such as:

* Age
* Time spent in hospital
* Previous outpatient visits
* Previous emergency visits
* Previous inpatient visits
* Number of diagnoses
* Admission type
* Discharge disposition
* Admission source
* Diagnosis information
* Readmission status

The original `readmitted` column contains:

```text
<30 → Readmitted within 30 days
>30 → Readmitted after 30 days
No  → No readmission
```

For this machine learning project, we convert it into a binary target:

```text
<30 → 1
>30 → 0
No  → 0
```

The source study used a final dataset of 69,984 encounters after applying its inclusion/exclusion procedure and selecting one encounter per patient for its logistic-regression analysis.

---

## 5. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Logistic Regression

---

## 6. Machine Learning Algorithm

### Logistic Regression

Logistic Regression is a classification algorithm used to predict the probability of an outcome.

In this project:

```text
Patient Information
        ↓
Logistic Regression
        ↓
Probability of Readmission
        ↓
0 or 1
```

Where:

```text
0 → Not readmitted within 30 days
1 → Readmitted within 30 days
```

The case study uses **L2 regularization** to help control model complexity.

---

## 7. Features Used

The project uses the following features:

```text
age
time_in_hospital
number_outpatient
number_emergency
number_inpatient
number_diagnoses
admission_type_id
discharge_disposition_id
admission_source_id
diag_1
diag_2
diag_3
```

### Numerical Features

```text
time_in_hospital
number_outpatient
number_emergency
number_inpatient
number_diagnoses
```

### Categorical Features

```text
age
admission_type_id
discharge_disposition_id
admission_source_id
diag_1
diag_2
diag_3
```

This is a simplified feature selection for the machine-learning implementation. It should not be treated as the exact final feature specification of the original research model.

---

## 8. Data Preprocessing

### 1. Load the dataset

```python
df = pd.read_csv("diabetic_data.csv")
```

### 2. Create the target variable

```python
df["target"] = df["readmitted"].apply(
    lambda x: 1 if x == "<30" else 0
)
```

This converts the original readmission categories into a binary classification target.

---

### 3. Separate features and target

```python
X = df[features]
y = df["target"]
```

Where:

```text
X → Patient information/features
y → Readmission outcome
```

---

### 4. Scale numerical features

```python
StandardScaler()
```

StandardScaler standardizes numerical features so that they are on a comparable scale.

---

### 5. Encode categorical features

```python
OneHotEncoder(handle_unknown="ignore")
```

Categorical variables cannot be directly given to Logistic Regression in their original text/category form.

One-Hot Encoding converts categories into numerical columns.

---

## 9. Train-Test Split

The dataset is divided into training and testing data:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The split is:

```text
80% → Training data
20% → Testing data
```

`stratify=y` helps maintain the target-class distribution in both sets.

---

## 10. Logistic Regression with L2 Regularization

The model is created using:

```python
model = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=1000
)
```

### `penalty="l2"`

Specifies **L2 regularization**.

Regularization helps control overly large model coefficients and can reduce overfitting.

### `C=1.0`

`C` controls the inverse of regularization strength.

```text
Smaller C → stronger regularization
Larger C  → weaker regularization
```

### `max_iter=1000`

Allows the optimization algorithm up to 1000 iterations to find the model parameters.

---

## 11. Pipeline

The preprocessing and model are combined into a single pipeline:

```python
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)
```

The pipeline performs:

```text
Raw Data
   ↓
Preprocessing
   ↓
Scaling + One-Hot Encoding
   ↓
Logistic Regression
   ↓
Prediction
```

This keeps the preprocessing and model steps together.

---

## 12. Model Training

The model is trained using:

```python
pipeline.fit(X_train, y_train)
```

The model learns patterns from the training data.

---

## 13. Prediction

### Class Prediction

```python
y_pred = pipeline.predict(X_test)
```

The model predicts:

```text
0 → Not readmitted within 30 days
1 → Readmitted within 30 days
```

### Probability Prediction

```python
y_prob = pipeline.predict_proba(X_test)[:, 1]
```

This gives the predicted probability of:

```text
Class 1 → Readmission within 30 days
```

These probabilities are used for ROC-AUC evaluation.

---

## 14. Model Evaluation

### ROC-AUC

```python
auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", auc)
```

ROC-AUC evaluates how well the model separates patients who are readmitted within 30 days from those who are not.

A higher ROC-AUC indicates better separation between the two classes.

---

### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

print(cm)
```

The confusion matrix contains:

```text
                 Predicted
              0          1

Actual 0      TN         FP
Actual 1      FN         TP
```

Where:

* **TN** → Correctly predicted no 30-day readmission
* **FP** → Predicted readmission but patient was not readmitted within 30 days
* **FN** → Patient was readmitted within 30 days but model predicted no readmission
* **TP** → Correctly predicted 30-day readmission

In this case study, false negatives are important because they represent 30-day readmissions that the model failed to identify.

---

## 15. Classification Report

```python
print(classification_report(y_test, y_pred))
```

The classification report provides:

* Precision
* Recall
* F1-score
* Support

### Precision

Answers:

> Of the patients predicted to be readmitted, how many actually were?

```text
Precision = TP / (TP + FP)
```

### Recall

Answers:

> Of the patients who were actually readmitted within 30 days, how many did the model identify?

```text
Recall = TP / (TP + FN)
```

### F1-score

Combines precision and recall into a single measure.

```text
F1 = 2 × (Precision × Recall)
          -------------------
          (Precision + Recall)
```

---

## 16. Project Workflow

```text
Hospital Patient Dataset
          ↓
     Load Dataset
          ↓
    Explore Dataset
          ↓
 Create 30-Day Target
          ↓
   Select Features
          ↓
Separate Numerical/Categorical
          ↓
     Preprocessing
    ↙             ↘
Scaling       One-Hot Encoding
    ↘             ↙
       Logistic Regression
       + L2 Regularization
              ↓
          Train Model
              ↓
           Predict
          ↙       ↘
    Class Output   Probability
        ↓              ↓
 Confusion Matrix    ROC-AUC
        ↓
Classification Report
```

---

## 17. Project Structure

```text
Hospital-Readmission-Prediction/
│
├── data/
│   └── diabetic_data.csv
│
├── hospital_readmission.py
│
├── requirements.txt
│
└── README.md
```

---

## 18. Requirements

Create a `requirements.txt` file:

```text
pandas
numpy
scikit-learn
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 19. How to Run the Project

### Step 1: Clone the repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Open the project folder

```bash
cd Hospital-Readmission-Prediction
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the program

```bash
python hospital_readmission.py
```

---

## 20. Expected Output

The program displays:

```text
First 5 rows
Dataset shape
Target distribution
Training data size
Testing data size
ROC-AUC
Confusion Matrix
Classification Report
```

Example format:

```text
ROC-AUC: <model result>

Confusion Matrix:
[[TN FP]
 [FN TP]]

Classification Report:
              precision    recall    f1-score
...
```

The actual values depend on the dataset and model run.

---

## 21. Important Medical Context

The original research examined the relationship between HbA1c measurement and hospital readmission among diabetic patient encounters. The study reported that HbA1c measurement was associated with readmission, but it explicitly noted that the observational analysis could not establish cause and effect.

Therefore, the machine-learning model in this project should be considered a **prediction/educational model**, not a medical diagnosis or a substitute for clinical decision-making.

---

## 22. Key Learning Points

This project demonstrates:

* Binary classification
* Logistic Regression
* L2 regularization
* Feature selection
* Numerical feature scaling
* One-Hot Encoding
* Pipeline
* Train-test split
* Stratified splitting
* Probability prediction
* ROC-AUC
* Confusion Matrix
* Precision
* Recall
* F1-score

---

## 23. Conclusion

This project demonstrates how machine learning can be used to predict whether a diabetic patient encounter is associated with **readmission within 30 days**.

A Logistic Regression model with **L2 regularization** is used along with preprocessing techniques such as StandardScaler and One-Hot Encoding.

The model is evaluated using ROC-AUC, confusion matrix, precision, recall, and F1-score.

The project provides a practical example of applying machine learning to hospital readmission prediction while highlighting the importance of careful interpretation of healthcare data.

"https://www.kaggle.com/code/chongchong33/predicting-hospital-readmission-of-diabetics/input?select=diabetic_data.csv" ---> Dataset Link

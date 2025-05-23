# Diabetes Classification Using Random Forest

This project builds a machine learning model to classify whether a person has diabetes based on biomedical data. The classification is performed using the **Random Forest** algorithm, and the dataset is preprocessed, visualized, and evaluated using standard data science techniques.

---

## 📁 Dataset

The dataset used in this project contains biomedical features and a target variable indicating diabetes status. It includes both numerical and categorical data.

- **Source**: [Provide dataset source or indicate if it's confidential/private]
- **Target Column**: `CLASS` (Yes/No)
- **Features include**:
  - Age
  - BMI
  - Blood Pressure
  - Glucose
  - Gender (F/M)
  - ...and other clinical measurements

---

## 🔍 Project Workflow

1. **Data Preprocessing**
   - Remove invalid entries (`CLASS == 'P'`)
   - Encode categorical data (e.g., Gender: F → 0, M → 1)
   - Strip whitespaces from target labels

2. **Exploratory Data Analysis**
   - Generate a heatmap to visualize correlation between features

3. **Modeling**
   - Train-test split (80/20)
   - Random Forest Classifier with `class_weight='balanced'` to handle imbalanced data

4. **Evaluation**
   - Accuracy score
   - Classification report (Precision, Recall, F1-Score)
   - Confusion matrix (visualized with seaborn)

---

## 📊 Example Outputs

### 🔸 Correlation Heatmap

Displays the relationship between features to understand feature importance and multicollinearity.

### 🔸 Confusion Matrix

Shows prediction vs actual classification outcomes.

---

## 📦 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

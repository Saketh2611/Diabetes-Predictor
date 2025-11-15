# 🧠 Diabetes Prediction using Autoencoders and Random Forest

This project compares two ML pipelines to predict diabetes from biomedical features:

1. **Baseline Model**: Random Forest Classifier (RFC) with hyperparameter tuning via GridSearchCV.
2. **Enhanced Model**: Autoencoder for dimensionality reduction, followed by RFC with GridSearchCV.

The goal is to show how representation learning with autoencoders can uncover hidden feature patterns and improve classification performance.

---

## 📁 Dataset

The dataset contains anonymized biomedical measurements and a binary target column indicating diabetes status.

- **Target**: `CLASS` (N = Non-diabetic, P = Diabetic)
- **Features include**:
  - `AGE`
  - `BMI`
  - `HbA1c`
  - `Cr` (Creatinine)
  - `Urea`, `TG`, `HDL`, `LDL`, `Chol`, etc.
  - `Gender` (F/M)

---

## 🔍 Project Workflow

1. **Data Preprocessing**
   - Removed ambiguous/invalid labels (`CLASS == 'P'`)
   - Encoded categorical variables (e.g., `Gender`: F → 0, M → 1)
   - Cleaned label formatting

2. **Exploratory Data Analysis**
   - Visualized feature correlations using a heatmap

3. **Modeling**
   - Train-test split (80/20)
   - RFC (with and without autoencoder)
   - Hyperparameter tuning using `GridSearchCV` with 5-fold cross-validation

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix (via seaborn)
   - Feature importance visualization

---

## ✅ Model 1: Random Forest Classifier (Baseline)

- Input: Raw preprocessed features
- Classifier: `RandomForestClassifier`
- Hyperparameter tuning: `GridSearchCV` with 5-fold cross-validation
  - Helps prevent **overfitting** by validating on multiple folds
- Output: Feature importance, accuracy, classification report

**Top Features Identified**:
- `HbA1c`
- `BMI`
- `AGE`

---

## 🚀 Model 2: Autoencoder + Random Forest

1. Trained an **autoencoder** on input features for 300 epochs
   - Compressed input to a **4D latent vector**
2. Used the **latent representation** as input to RFC
3. Tuned RFC using `GridSearchCV` (5-fold CV)
4. Compared performance and feature importances

> The autoencoder revealed non-obvious latent interactions that boosted performance.

---

## 📈 Results Comparison

| Model            | Accuracy |    Notes                             |
|------------------|----------|------------------------|-----------------|
| RFC Baseline     | ~97%     |  Based on raw input features         |
| AE + RFC         | ~97%     |  Autoencoder discovered new patterns |

---

## 📊 Example Outputs

- **Correlation Heatmap**: Understands feature relationships
- **Confusion Matrix**: Validates classification performance
- **Feature Importance Plots**: Compared before vs after AE

---

## 📁 Notebooks

Open either notebook to explore the full workflow:

- 🔹 `1_RFC_Baseline.ipynb`: Raw features + RFC + GridSearchCV
- 🔹 `2_Autoencoder_RFC.ipynb`: Autoencoder (PyTorch) + RFC + GridSearchCV

---

## 🛠️ Tech Stack

- 🐍 Python 3.10+
- 🔥 PyTorch (for Autoencoder)
- 🌲 Scikit-learn (RFC + GridSearchCV)
- 📊 Matplotlib, Seaborn
- 🧮 Pandas, NumPy
- Joblib

---

## 📌 Key Learnings

- Autoencoders can **restructure feature space**, enabling models to learn deeper interactions
- `GridSearchCV` with cross-validation is essential to **reduce overfitting risk**
- Even classical models like RFC benefit from better input representations

---

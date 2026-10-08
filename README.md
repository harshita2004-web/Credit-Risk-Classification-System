# 💳 Credit Risk Classification System

A Machine Learning system that predicts whether a loan applicant is likely to **default** or **repay** their loan, based on their financial and personal details — helping lenders make smarter, data-driven credit decisions.

## 🎯 What It Does

Given details about a loan applicant (age, income, loan amount, employment history, credit history, etc.), the system predicts the **credit risk** — whether the person is likely to default on the loan or not — along with a probability score.

## 🛠️ Tech Stack

- **Python** — core language
- **Pandas** — data loading and preprocessing
- **Scikit-learn** — machine learning models and evaluation
- **Logistic Regression, KNN, Random Forest** — classification algorithms
- **GridSearchCV** — hyperparameter tuning

## ⚙️ How It Works

1. **Data Loading** — Dataset of 32,581 loan applicants loaded from CSV
2. **Data Cleaning** — Missing values in employment length and interest rate filled using median imputation
3. **Encoding** — Categorical features (home ownership, loan intent, loan grade, default history) converted to numeric using Label Encoding
4. **Train-Test Split** — Data split into 80% training and 20% testing sets
5. **Model Training** — Three models trained and compared: Logistic Regression, KNN, and Random Forest
6. **Hyperparameter Tuning** — GridSearchCV used to find the optimal Random Forest configuration
7. **Evaluation** — Models evaluated using Accuracy, Confusion Matrix, Classification Report, and ROC-AUC Score

## 📊 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 82.9% | 0.673 |
| KNN | 87.8% | 0.774 |
| Random Forest | 93.0% | 0.851 |
| Random Forest (Tuned via GridSearch) | 92.8% | 0.846 |

**Best Parameters (GridSearch):** `n_estimators=200, max_depth=20, min_samples_split=5`

Random Forest significantly outperformed the other models, especially in correctly identifying high-risk (defaulting) applicants — critical for real-world credit risk assessment.

## 📁 Project Structure
├── credit_risk_dataset.csv
├── main.py # Data loading & exploration
├── split_data.py # Feature/target split
├── train_test_split.py # Train-test split
├── logistic_regression_model.py # Logistic Regression model
├── knn_model.py # KNN model
├── random_forest_model.py # Random Forest model
├── gridsearch_rf.py # Hyperparameter tuning
└── test_prediction.py # Real-time prediction test

## 🚀 How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python random_forest_model.py
```
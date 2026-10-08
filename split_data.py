import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Dataset load karna
df = pd.read_csv("credit_risk_dataset.csv")

# Missing values ko median se fill karna
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# Categorical columns ko number mein convert karna
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features (X) aur Target (y) alag karna
X = df.drop('loan_status', axis=1)
y = df['loan_status']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
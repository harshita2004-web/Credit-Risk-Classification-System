import pandas as pd

# Dataset load karna
df = pd.read_csv("credit_risk_dataset.csv")

# Pehli 5 rows dekhna
print(df.head())

# Dataset ka shape (kitni rows, kitne columns)
print("Shape:", df.shape)

# Column names aur unka data type
print(df.info())

# Missing values check karna
print(df.isnull().sum())

# Missing values ko median se fill karna
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# Check karna ki missing values khatam ho gaye
print(df.isnull().sum())
from sklearn.preprocessing import LabelEncoder

# Categorical columns ki list
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Har categorical column ko number mein convert karna
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Check karna ki sab kuch number ban gaya
print(df.dtypes)
print(df.head())
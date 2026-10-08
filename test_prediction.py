import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Dataset load karna
df = pd.read_csv("credit_risk_dataset.csv")

# Missing values ko median se fill karna
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# Categorical columns ko number mein convert karna
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features (X) aur Target (y) alag karna
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Model train karna (best settings ke saath)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

# ============================
# Ek NAYA customer ka sample data (test karne ke liye)
# ============================
new_customer = pd.DataFrame({
    'person_age': [25],
    'person_income': [50000],
    'person_home_ownership': [encoders['person_home_ownership'].transform(['RENT'])[0]],
    'person_emp_length': [3.0],
    'loan_intent': [encoders['loan_intent'].transform(['EDUCATION'])[0]],
    'loan_grade': [encoders['loan_grade'].transform(['B'])[0]],
    'loan_amnt': [10000],
    'loan_int_rate': [11.5],
    'loan_percent_income': [0.20],
    'cb_person_default_on_file': [encoders['cb_person_default_on_file'].transform(['N'])[0]],
    'cb_person_cred_hist_length': [4]
})

# Prediction karna
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

print("Prediction (0 = No Default, 1 = Default):", prediction[0])
print("Probability [No Default, Default]:", probability[0])

if prediction[0] == 1:
    print("Result: Ye customer LOAN DEFAULT kar sakta hai — Risky!")
else:
    print("Result: Ye customer safe hai — Loan default nahi karega.")
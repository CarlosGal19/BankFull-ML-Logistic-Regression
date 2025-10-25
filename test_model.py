import pandas as pd
import joblib

model = joblib.load("logistic_model.pkl")  # tu modelo guardado
scaler = joblib.load("scaler.pkl")         # si escalaste los datos
test_data = [
    {
        "age": 58, "education": 3, "default": 0, "balance": 2143, "housing": 1, "loan": 0,
        "day": 5, "duration": 261, "campaign": 1, "pdays": -1, "previous": 0, "Target": 0,
        "job_blue-collar": False, "job_entrepreneur": False, "job_housemaid": False, "job_management": True,
        "job_retired": False, "job_self-employed": False, "job_services": False, "job_student": False,
        "job_technician": False, "job_unemployed": False, "job_unknown": False,
        "marital_married": True, "marital_single": False,
        "contact_telephone": False, "contact_unknown": True,
        "month_aug": False, "month_dec": False, "month_feb": False, "month_jan": False,
        "month_jul": False, "month_jun": False, "month_mar": False, "month_may": True,
        "month_nov": False, "month_oct": False, "month_sep": False,
        "poutcome_other": False, "poutcome_success": False, "poutcome_unknown": True
    },
    {
        "age": 44, "education": 2, "default": 0, "balance": 29, "housing": 1, "loan": 0,
        "day": 5, "duration": 151, "campaign": 1, "pdays": -1, "previous": 0, "Target": 0,
        "job_blue-collar": False, "job_entrepreneur": False, "job_housemaid": False, "job_management": False,
        "job_retired": False, "job_self-employed": False, "job_services": False, "job_student": False,
        "job_technician": True, "job_unemployed": False, "job_unknown": False,
        "marital_married": False, "marital_single": True,
        "contact_telephone": False, "contact_unknown": True,
        "month_aug": False, "month_dec": False, "month_feb": False, "month_jan": False,
        "month_jul": False, "month_jun": False, "month_mar": False, "month_may": True,
        "month_nov": False, "month_oct": False, "month_sep": False,
        "poutcome_other": False, "poutcome_success": False, "poutcome_unknown": True
    },
    {
        "age": 33, "education": 2, "default": 0, "balance": 2, "housing": 1, "loan": 1,
        "day": 5, "duration": 76, "campaign": 1, "pdays": -1, "previous": 0, "Target": 0,
        "job_blue-collar": False, "job_entrepreneur": True, "job_housemaid": False, "job_management": False,
        "job_retired": False, "job_self-employed": False, "job_services": False, "job_student": False,
        "job_technician": False, "job_unemployed": False, "job_unknown": False,
        "marital_married": True, "marital_single": False,
        "contact_telephone": False, "contact_unknown": True,
        "month_aug": False, "month_dec": False, "month_feb": False, "month_jan": False,
        "month_jul": False, "month_jun": False, "month_mar": False, "month_may": True,
        "month_nov": False, "month_oct": False, "month_sep": False,
        "poutcome_other": False, "poutcome_success": False, "poutcome_unknown": True
    },
    {
        "age": 47, "education": 0, "default": 0, "balance": 1506, "housing": 1, "loan": 0,
        "day": 5, "duration": 92, "campaign": 1, "pdays": -1, "previous": 0, "Target": 0,
        "job_blue-collar": True, "job_entrepreneur": False, "job_housemaid": False, "job_management": False,
        "job_retired": False, "job_self-employed": False, "job_services": False, "job_student": False,
        "job_technician": False, "job_unemployed": False, "job_unknown": False,
        "marital_married": True, "marital_single": False,
        "contact_telephone": False, "contact_unknown": True,
        "month_aug": False, "month_dec": False, "month_feb": False, "month_jan": False,
        "month_jul": False, "month_jun": False, "month_mar": False, "month_may": True,
        "month_nov": False, "month_oct": False, "month_sep": False,
        "poutcome_other": False, "poutcome_success": False, "poutcome_unknown": True
    },
    {
        "age": 29, "education": 1, "default": 0, "balance": 0, "housing": 1, "loan": 0,
        "day": 12, "duration": 120, "campaign": 2, "pdays": -1, "previous": 0, "Target": 0,
        "job_blue-collar": False, "job_entrepreneur": False, "job_housemaid": True, "job_management": False,
        "job_retired": False, "job_self-employed": False, "job_services": False, "job_student": False,
        "job_technician": False, "job_unemployed": False, "job_unknown": False,
        "marital_married": False, "marital_single": True,
        "contact_telephone": False, "contact_unknown": True,
        "month_aug": False, "month_dec": False, "month_feb": False, "month_jan": False,
        "month_jul": False, "month_jun": False, "month_mar": False, "month_may": True,
        "month_nov": False, "month_oct": False, "month_sep": False,
        "poutcome_other": False, "poutcome_success": False, "poutcome_unknown": True
    },
    {
        "age": 30, "education": 3, "default": 0, "balance": 12000, "housing": 1, "loan": 0,
        "day": 5, "duration": 1500, "campaign": 2, "pdays": -1, "previous": 0, "Target": 1,
        "job_blue-collar": False, "job_entrepreneur": False, "job_housemaid": False, "job_management": False,
        "job_retired": False, "job_self-employed": False, "job_services": False, "job_student": False,
        "job_technician": False, "job_unemployed": True, "job_unknown": False,
        "marital_married": True, "marital_single": False,
        "contact_telephone": False, "contact_unknown": True,
        "month_aug": False, "month_dec": False, "month_feb": False, "month_jan": False,
        "month_jul": False, "month_jun": False, "month_mar": False, "month_may": False,
        "month_nov": False, "month_oct": False, "month_sep": False,
        "poutcome_other": False, "poutcome_success": False, "poutcome_unknown": False
    }
]

df_test = pd.DataFrame(test_data)

if "Target" in df_test.columns:
    df_test_features = df_test.drop("Target", axis=1)
else:
    df_test_features = df_test

df_scaled = scaler.transform(df_test_features)

probs = model.predict_proba(df_scaled)[:,1]

for i, prob in enumerate(probs, start=1):
    print(f"Registro {i}: Probabilidad = {prob:.2f}")

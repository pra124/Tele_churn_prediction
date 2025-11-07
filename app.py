from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("grad_boost.pkl", "rb"))
scaler = pickle.load(open("stand_scalar.pkl", "rb"))

# ✅ Get feature list from model itself
final_cols = model.feature_names_in_

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ------- 1. Read Input -------
        data = {
            'gender': request.form['gender'],
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'Contract': request.form['Contract'],
            'sim': request.form['sim'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'JoinYear': int(request.form['JoinYear']),
            'MonthlyCharges_qan_quantiles': float(request.form['MonthlyCharges']),
            'TotalCharges_KNN_imp_qan_quantiles': float(request.form['TotalCharges'])
        }

        df = pd.DataFrame([data])

        # ------- 2. One-Hot Encoding -------
        cat_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService',
                    'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                    'StreamingMovies','PaperlessBilling','PaymentMethod']

        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # ------- 3. Ordinal Encoding for Contract -------
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        df['Contract_con'] = contract_map.get(df['Contract'][0], 0)
        df.drop('Contract', axis=1, inplace=True)

        # ------- 4. Map sim -------
        sim_map = {'Jio': 0, 'Airtel': 1, 'Vi': 2, 'BSNL': 3}
        df['sim'] = sim_map.get(df['sim'][0], 0)

        # ------- 5. Scale -------
        df[['MonthlyCharges_qan_quantiles','TotalCharges_KNN_imp_qan_quantiles']] = scaler.transform(
            df[['MonthlyCharges_qan_quantiles','TotalCharges_KNN_imp_qan_quantiles']]
        )

        # ✅ -------- 6. Handle Missing Columns Automatically --------
        for col in final_cols:
            if col not in df.columns:
                df[col] = 0

        # ✅ Keep exact order expected by model
        df = df[final_cols]

        # ✅ 7. Predict
        pred = model.predict(df)[0]
        result = "Customer Will Churn ❌" if pred == 1 else "Customer Will Stay ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
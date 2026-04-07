import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
OUTPUT_MARKDOWN = BASE_DIR / "model_performance_report.md"

def df_to_markdown(df):
    """Simple manual markdown table generator to avoid tabulate dependency."""
    if df.empty:
        return ""
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join([str(val) for val in row]) + " |")
    return "\n".join(lines) + "\n"

def evaluate_main_insurance():
    report_lines = ["# Model Performance Report\n\n"]
    report_lines.append("## Main Insurance Dataset Evaluation (Claims)\n\n")
    
    data_path = BASE_DIR / "insurance.csv"
    if not data_path.exists():
        return report_lines + ["Main insurance.csv not found.\n\n"]

    df = pd.read_csv(data_path)
    
    numerical_cols = ['age', 'bmi', 'bloodpressure', 'children', 'No_Claim_Years']
    feature_cols = ['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker', 'No_Claim_Years']
    target_col = 'claim'

    try:
        le_gender = joblib.load(BASE_DIR / "label_encoder_gender.pkl")
        le_smoker = joblib.load(BASE_DIR / "label_encoder_smoker.pkl")
        le_diabetic = joblib.load(BASE_DIR / "label_encoder_diabetic.pkl")
        scaler = joblib.load(BASE_DIR / "scaler.pkl")
        poly = joblib.load(BASE_DIR / "poly_features.pkl")
        imputer = joblib.load(BASE_DIR / "imputer.pkl")
    except Exception as e:
        return report_lines + [f"Error loading artifacts: {e}\n\n"]

    df['gender'] = le_gender.transform(df['gender'])
    df['smoker'] = le_smoker.transform(df['smoker'])
    if 'diabetic' in df.columns:
        df['diabetic'] = le_diabetic.transform(df['diabetic'])
    else:
        df['diabetic'] = 0

    df[numerical_cols] = imputer.transform(df[numerical_cols])
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    X_raw = df[feature_cols]
    poly_features = poly.transform(X_raw)
    poly_feature_names = poly.get_feature_names_out(feature_cols)
    X = pd.DataFrame(poly_features, columns=poly_feature_names)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_files = {
        "Linear Regression": "model_lr.pkl",
        "Decision Tree": "model_dt.pkl",
        "Random Forest": "model_rf.pkl",
        "Gradient Boosting": "model_gb.pkl",
        "K-Nearest Neighbors": "model_knn.pkl",
        "Support Vector Regressor": "model_svr.pkl",
        "XGBoost": "model_xgb.pkl"
    }

    results = []
    sample_preds_data = None

    for name, filename in models_files.items():
        try:
            model = joblib.load(BASE_DIR / filename)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            results.append({"Model": name, "R2 Score": f"{r2:.4f}", "MAE": f"{mae:.2f}"})
            
            if name == "Gradient Boosting":
                sample_preds_data = pd.DataFrame({"Actual": y_test[:5].values, "Predicted": [round(v, 2) for v in y_pred[:5]]})
        except:
            continue

    if results:
        df_results = pd.DataFrame(results)
        report_lines.append("### Regression Metrics\n\n")
        report_lines.append(df_to_markdown(df_results) + "\n")
        
        if sample_preds_data is not None:
            report_lines.append("### Sample Predictions (Gradient Boosting)\n\n")
            report_lines.append(df_to_markdown(sample_preds_data) + "\n")

    try:
        clf = joblib.load(BASE_DIR / "risk_classifier.pkl")
        y_class = (y > 15000).astype(int)
        _, X_test_c, _, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
        acc = accuracy_score(y_test_c, clf.predict(X_test_c))
        report_lines.append(f"### Classification Accuracy\n\n- **Risk Classifier (Random Forest):** {acc:.4f} accuracy\n\n")
    except:
        pass

    return report_lines

def evaluate_domain_models():
    report_lines = ["## Domain-Specific Models Evaluation\n\n"]
    domain_files = [
        ("Business", "business_insurance_10000.csv", "business_models.pkl"),
        ("Health", "health_insurance_10000.csv", "health_models.pkl"),
        ("Life", "life_insurance_1000.csv", "life_models.pkl"),
        ("Motor", "motor_insurance_10000.csv", "motor_models.pkl"),
        ("Property", "property_insurance_10000.csv", "property_models.pkl"),
        ("Specialty", "specialty_insurance_10000.csv", "specialty_models.pkl"),
        ("Travel", "travel_insurance_10000.csv", "travel_models.pkl"),
        ("Claim (Domain)", "insurance.csv", "claim_models.pkl")
    ]

    all_domain_results = []

    for domain, csv, pkl in domain_files:
        csv_path = BASE_DIR / csv
        pkl_path = BASE_DIR / pkl
        
        if not csv_path.exists() or not pkl_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        
        if csv == "insurance.csv":
            target_col = "claim"
            drop_cols = ["Id"]
        else:
            if 'Premium_Amount' in df.columns: target_col = 'Premium_Amount'
            elif 'Premium' in df.columns and 'Annual_Premium' not in df.columns: target_col = 'Premium'
            elif 'Annual_Premium' in df.columns: target_col = 'Annual_Premium'
            else: continue
            drop_cols = ['Policy_ID', 'Claim_Status']

        y = df[target_col]
        X = df.drop(columns=[col for col in drop_cols + [target_col] if col in df.columns])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        try:
            pipelines = joblib.load(pkl_path)
            for name, pipeline in pipelines.items():
                y_pred = pipeline.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                all_domain_results.append({
                    "Domain": domain,
                    "Model": name,
                    "R2 Score": f"{r2:.4f}"
                })
        except:
            continue

    if all_domain_results:
        df_res = pd.DataFrame(all_domain_results)
        pivot_df = df_res.pivot(index='Domain', columns='Model', values='R2 Score').reset_index()
        report_lines.append("### Domain R2 Scores Summary\n\n")
        report_lines.append(df_to_markdown(pivot_df) + "\n")

    return report_lines

if __name__ == "__main__":
    full_report = evaluate_main_insurance()
    full_report += evaluate_domain_models()
    
    with open(OUTPUT_MARKDOWN, "w", encoding="utf-8") as f:
        f.writelines(full_report)
    print(f"Report generated: {OUTPUT_MARKDOWN}")

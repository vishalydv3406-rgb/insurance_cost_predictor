import pandas as pd
import numpy as np
import joblib
import json
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent

def process_dataset(filepath):
    df = pd.read_csv(filepath)
    filename = Path(filepath).name
    
    if filename == "insurance.csv":
        insurance_type = "Claim"
        target_col = "claim"
        drop_cols = ["Id"]
    else:
        insurance_type = filename.split('_insurance')[0].capitalize()

        # Determine target
        if 'Premium_Amount' in df.columns:
            target_col = 'Premium_Amount'
        elif 'Premium' in df.columns and 'Annual_Premium' not in df.columns:
            target_col = 'Premium'
        elif 'Annual_Premium' in df.columns:
            target_col = 'Annual_Premium'
        else:
            print(f"Skipping {filename}, target column not found.")
            return
        drop_cols = ['Policy_ID', 'Claim_Status'] 
    
    y = df[target_col]
    X = df.drop(columns=[col for col in drop_cols + [target_col] if col in df.columns])

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Create config for UI
    config = {
        "insurance_type": insurance_type,
        "target_col": target_col,
        "features": {
            "numerical": [],
            "categorical": {}
        }
    }

    for col in numerical_cols:
        config["features"]["numerical"].append({
            "name": col,
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean())
        })

    for col in categorical_cols:
        cats = X[col].fillna("Unknown").astype(str).unique().tolist()
        config["features"]["categorical"][col] = cats

    # Preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        # Using handle_unknown='ignore' so unseen categories don't break predictions
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    models = {
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trained_pipelines = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', model)])
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        print(f"{insurance_type} - {name} R2 Score: {score:.3f}")
        trained_pipelines[name] = pipeline

    # Save format: {insurance_type}_models.pkl
    name_lower = insurance_type.lower()
    joblib.dump(trained_pipelines, BASE_DIR / f"{name_lower}_models.pkl")
    with open(BASE_DIR / f"{name_lower}_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
if __name__ == '__main__':
    csv_files = glob.glob(str(BASE_DIR / '*_insurance_*.csv'))
    csv_files.append(str(BASE_DIR / 'insurance.csv'))
    for f in csv_files:
        print(f"Processing {Path(f).name}...")
        try:
            process_dataset(f)
        except Exception as e:
            print(f"Error on {f}: {e}")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "insurance.csv"
MODEL_PATH = BASE_DIR / "best_model.pkl"
MODEL_LR_PATH = BASE_DIR / "model_lr.pkl"
MODEL_DT_PATH = BASE_DIR / "model_dt.pkl"
MODEL_RF_PATH = BASE_DIR / "model_rf.pkl"
MODEL_GB_PATH = BASE_DIR / "model_gb.pkl"
MODEL_KNN_PATH = BASE_DIR / "model_knn.pkl"
MODEL_SVR_PATH = BASE_DIR / "model_svr.pkl"
MODEL_XGB_PATH = BASE_DIR / "model_xgb.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
POLY_PATH = BASE_DIR / "poly_features.pkl"
IMPUTER_PATH = BASE_DIR / "imputer.pkl"
LE_GENDER_PATH = BASE_DIR / "label_encoder_gender.pkl"
LE_SMOKER_PATH = BASE_DIR / "label_encoder_smoker.pkl"
LE_DIABETIC_PATH = BASE_DIR / "label_encoder_diabetic.pkl"

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # Removing dropna() to explicitly learn how to impute missing data as an FDS concept
    return df

def train():
    print("re-training model with corrected feature handling...")
    df = load_data()
    
    # Define Column Groups
    numerical_cols = ['age', 'bmi', 'bloodpressure', 'children', 'No_Claim_Years']
    categorical_cols = ['gender', 'smoker', 'diabetic']
    feature_cols = ['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker', 'No_Claim_Years']
    target_col = 'claim'
    
    # 1. Encoders
    le_gender = LabelEncoder()
    le_smoker = LabelEncoder()
    le_diabetic = LabelEncoder()
    
    df['gender'] = le_gender.fit_transform(df['gender'])
    df['smoker'] = le_smoker.fit_transform(df['smoker'])
    
    if 'diabetic' in df.columns:
        df['diabetic'] = le_diabetic.fit_transform(df['diabetic'])
    else:
        # Fallback if diabetic column is missing in source but required by app logic
        # Ideally we should fix data source, but for now lets assume it exists or handle it
        print("Warning: 'diabetic' column not found in dataset. Creating dummy column for compatibility.")
        df['diabetic'] = 0 
        le_diabetic.fit([0, 1]) # Fit on dummy values to ensure encoder works

    # 2. Imputation & Scaling - ONLY fit on numerical columns
    # FDS Concept: We use median imputation because mean is sensitive to outliers
    imputer = SimpleImputer(strategy='median')
    df_imputed = df.copy()
    df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    scaler = StandardScaler()
    scaler.fit(df_imputed[numerical_cols])
    
    # Transform numerical columns in place or create a copy
    df_scaled = df_imputed.copy()
    df_scaled[numerical_cols] = scaler.transform(df_imputed[numerical_cols])
    
    # 3. Polynomial Features (Interaction Terms)
    # We want to catch interactions like bmi * smoker
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_raw = df_scaled[feature_cols]
    
    # Fit poly only on select columns to avoid feature explosion
    poly_features = poly.fit_transform(X_raw)
    
    # We recreate a dataframe so models can track feature names if they want
    poly_feature_names = poly.get_feature_names_out(feature_cols)
    X = pd.DataFrame(poly_features, columns=poly_feature_names)
    y = df_scaled[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Setup Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print(f"Linear Regression - MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}, R2: {r2_score(y_test, y_pred_lr):.2f}")

    # Model 2: Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print(f"Decision Tree - MAE: {mean_absolute_error(y_test, y_pred_dt):.2f}, R2: {r2_score(y_test, y_pred_dt):.2f}")

    # Model 3: Random Forest (w/ GridSearchCV)
    print("Tuning Random Forest...")
    rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=cv, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf = rf_grid.best_estimator_
    y_pred_rf = rf.predict(X_test)
    print(f"Random Forest (Tuned) - Best Params: {rf_grid.best_params_} | MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}, R2: {r2_score(y_test, y_pred_rf):.2f}")

    # Model 4: Gradient Boosting (w/ GridSearchCV)
    print("Tuning Gradient Boosting...")
    gb_param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]}
    gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_param_grid, cv=cv, scoring='r2', n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    gb = gb_grid.best_estimator_
    y_pred_gb = gb.predict(X_test)
    print(f"Gradient Boosting (Tuned) - Best Params: {gb_grid.best_params_} | MAE: {mean_absolute_error(y_test, y_pred_gb):.2f}, R2: {r2_score(y_test, y_pred_gb):.2f}")
    
    # Model 5: K-Nearest Neighbors
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print(f"K-Nearest Neighbors - MAE: {mean_absolute_error(y_test, y_pred_knn):.2f}, R2: {r2_score(y_test, y_pred_knn):.2f}")

    # Model 6: Support Vector Regressor
    print("Training Support Vector Regressor (on subsample)...")
    svr = SVR(kernel='rbf', C=1000, gamma='scale')
    # Subsample for SVR since it's O(N^3)
    sample_size = min(2000, len(X_train))
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    svr.fit(X_train.iloc[idx], y_train.iloc[idx])
    y_pred_svr = svr.predict(X_test)
    print(f"Support Vector Regressor - MAE: {mean_absolute_error(y_test, y_pred_svr):.2f}, R2: {r2_score(y_test, y_pred_svr):.2f}")

    # Model 7: XGBoost
    print("Training XGBoost...")
    xgb_reg = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    xgb_reg.fit(X_train, y_train)
    y_pred_xgb = xgb_reg.predict(X_test)
    print(f"XGBoost - MAE: {mean_absolute_error(y_test, y_pred_xgb):.2f}, R2: {r2_score(y_test, y_pred_xgb):.2f}")

    # We will log best_model for classification usage, but save all 7 separately.
    best_model = gb if r2_score(y_test, y_pred_gb) > r2_score(y_test, y_pred_rf) else rf

    # --- NEW: Train Classification Model (High Risk vs Low Risk) ---
    # Threshold: Claim > 15000 is considered High Risk
    print("Training Risk Classifier...")
    y_class = (y > 15000).astype(int) # 1 if High Risk, 0 if Low Risk
    
    # Use Random Forest for classification
    from sklearn.ensemble import RandomForestClassifier
    # Split for classification
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_c, y_train_c)
    
    accuracy = clf.score(X_test_c, y_test_c)
    print(f"Risk Classifier Accuracy: {accuracy:.2f}")

    # Save all artifacts
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(lr, MODEL_LR_PATH)
    joblib.dump(dt, MODEL_DT_PATH)
    joblib.dump(rf, MODEL_RF_PATH)
    joblib.dump(gb, MODEL_GB_PATH)
    joblib.dump(knn, MODEL_KNN_PATH)
    joblib.dump(svr, MODEL_SVR_PATH)
    joblib.dump(xgb_reg, MODEL_XGB_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(poly, POLY_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(le_gender, LE_GENDER_PATH)
    joblib.dump(le_smoker, LE_SMOKER_PATH)
    joblib.dump(le_diabetic, LE_DIABETIC_PATH)
    joblib.dump(clf, BASE_DIR / "risk_classifier.pkl") # Save classifier
        
    print("Model and artifacts saved successfully.")

if __name__ == "__main__":
    train()

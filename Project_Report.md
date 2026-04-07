# Insurance AI Pro+: Data Science Project Lifecycle Report

This document outlines the complete end-to-end Data Science lifecycle applied to the **Insurance AI Pro+** (Fundamentals of Data Science) project. The project demonstrates the implementation of predictive modeling across various insurance domains using an interactive, AI-driven dashboard.

---

## 1️⃣ Problem Definition
**Objective:** The goal is to build an intelligent, multi-domain insurance quotation system capable of accurately predicting insurance premiums or claim amounts based on specific demographic, physical, and historical user attributes. 

Furthermore, the system aims to provide **Explainable AI (XAI)** insights, offering users data-driven criteria to reduce their premium costs and maximize their claim approvals.

**ML Type:** 
- **Regression:** Estimating continuous numerical values (Premium Cost / Claim Amount).
- **Classification:** Categorizing overall candidate profiles as "High Risk" vs "Low Risk".

---

## 2️⃣ Data Collection
**Process:** Since real-world actuarial data is highly proprietary, the data was gathered via custom statistical data generation (`generate_data.py`). Data was generated strictly mirroring real-world epidemiological and physical correlations rather than pure uniform randomness.

**Multiple Domains:** Generated datasets spanning 8 distinct categories (Health, Life, Motor, Business, Property, Travel, Specialty, and core Claims), totaling over 120,000 records.

👉 **Tools Used:** `Python`, `numpy`, `pandas`

---

## 3️⃣ Data Understanding (Exploratory Data Analysis - EDA)
**Process:** Analyzed the structural integrity and distribution patterns of the generated features. 
- **Distributions:** Confirmed that structural variables match reality (e.g., BMI follows a Normal Distribution centered at 28; Number of Children follows a Poisson Distribution heavily weighted between 0-2).
- **Interactive EDA:** An interactive *Data Explorer* was built into the web application, providing dynamic histograms and correlation matrices.
- **Automated Static EDA:** A dedicated pipeline (`generate_eda_reports.py`) was developed to automatically iterate through all 8 datasets, generating over 40+ static visualizations (Target distributions, Correlation maps, and NCB impact boxes) stored in the `eda_outputs/` directory for deep actuarial audit.

👉 **Techniques:** Summary Statistics, Scatter Matrices, Correlation Heatmaps, Box-Whisker Plots.  
👉 **Tools Used:** `pandas`, `plotly.express`, `matplotlib`, `seaborn`, `streamlit`

---

## 4️⃣ Data Cleaning
**Process:** The raw datasets contained intentionally injected null values (an FDS concept to simulate dirty data) and required structured handling before training.

- **Missing Values (Numerical):** Handled using a `SimpleImputer` equipped with a `median` strategy, preventing extreme outliers from skewing the data distribution.
- **Missing Values (Categorical):** Handled using a `constant` strategy to label missing categories universally as "Unknown".
- **Outliers:** `np.clip()` was utilized during generation and preprocessing pipelines to cap mathematically impossible bounds (e.g., clipping Age to sensible adult values, capping maximum Poisson distributions for dependents).

👉 **Tools Used:** `scikit-learn` (`SimpleImputer`), `pandas`

---

## 5️⃣ Feature Engineering
**Process:** Transforming raw independent variables into a robust configuration optimized for mathematically-driven machine learning models.

- **Encoding Categorical Variables:** 
  - `LabelEncoder` was deployed for binary classification strings (e.g., Gender, Smoker).
  - `OneHotEncoder` (with `handle_unknown='ignore'`) was built into the inference pipelines for larger multi-class variables (e.g., Region, Insurance Sub-Type).
- **Feature Scaling:** `StandardScaler` was applied globally across numerical features (Age, Blood Pressure, BMI, Coverage Amounts) to normalize variance and force means toward 0, optimizing distance-based algorithms like KNN and SVR.
- **Interaction Creation:** `PolynomialFeatures` (degree=2) was introduced to automatically combine variables, specifically capturing synergistic risk factors (e.g., the compound actuarial risk of High BMI *and* Smoking concurrently).
- **Loyalty/NCB Feature:** A `No_Claim_Years` feature (0–5 years) was engineered across all datasets, applying a calculated **2% compounding discount** per claim-free year to the target premium/claim amount, mirroring modern insurance incentive structures.

👉 **Tools Used:** `scikit-learn.preprocessing`, `numpy`

---

## 6️⃣ Data Splitting
**Process:** Data strict isolation to properly validate model generalization.

- **Split Ratio:** The massive 50,000-row generic dataset, alongside the domain-specific 10,000-row datasets, were split strictly into an **80% Training Set** and a **20% Test Set**. Random states were fixed universally (`random_state=42`) ensuring reproducibility across runs.

👉 **Tools Used:** `scikit-learn.model_selection` (`train_test_split`)

---

## 7️⃣ Model Selection
**Process:** Multiple algorithms were selected reflecting varying degrees of systemic complexity.

| Problem Profile | Chosen Models |
| :--- | :--- |
| **Simple Linear Regression** | Linear Regression, Support Vector Regressor (SVR) |
| **Non-Linear / Distance-Based** | Decision Tree Regressor, K-Nearest Neighbors (KNN) |
| **Complex Ensemble (Trees)** | Random Forest, Gradient Boosting, XGBoost |
| **Risk Thresholding (Classification)** | Random Forest Classifier |

**Choice Logic:** Premium calculations are inherently non-linear and conditional (e.g., a smoker penalty acts as a multiplier, not a flat addition). Hence, ensemble tree algorithms like **Gradient Boosting** were hypothesized to perform best.

👉 **Tools Used:** `scikit-learn.ensemble`, `xgboost`, `scikit-learn.svm`

---

## 8️⃣ Model Training
**Process:** The algorithms were fitted across the preprocessed data matrices.
- The pipeline securely bundled imputation matrices, encoding logic, and scalar definitions alongside the ultimate regression model to prevent data leakage.
- *Computational Scaling:* For $O(N^3)$ models like SVR, randomized sub-sampling operations were invoked specifically to bypass excessive quadratic training time loops on the 50K row datasets via NumPy.

👉 **Code Example:** `model.fit(X_train, y_train)`

---

## 9️⃣ Model Evaluation
**Process:** We evaluated performance independently leveraging strict testing datasets.

---

## 9️⃣ Model Evaluation
**Process:** We evaluated performance independently leveraging strict testing datasets (20% holdout).

### Regression Metrics (Claims Dataset)

| Algorithm | $R^2$ Score | MAE (Mean Absolute Error) |
| :--- | :--- | :--- |
| **Gradient Boosting** | **0.9362** | **$473.12** |
| **XGBoost** | 0.9360 | $472.40 |
| **Random Forest** | 0.9288 | $500.80 |
| **Linear Regression** | 0.9287 | $508.48 |
| **K-Nearest Neighbors** | 0.9012 | $558.95 |
| **Decision Tree** | 0.8641 | $684.06 |
| **Support Vector (SVR)** | 0.8421 | $632.41 |

### Domain-Specific Accuracy ($R^2$ Score)

| Domain | Gradient Boosting | Linear Regression | Random Forest |
| :--- | :--- | :--- | :---|
| **Travel** | **0.7209** | 0.6505 | 0.6959 |
| **Life** | **0.6828** | 0.6430 | 0.6486 |
| **Health** | **0.6576** | 0.5899 | 0.6201 |
| **Motor** | **0.6301** | 0.5725 | 0.5963 |
| **Property** | 0.6198 | **0.6253** | 0.5823 |
| **Business** | **0.5967** | 0.5374 | 0.5701 |
| **Specialty** | **0.5927** | 0.5283 | 0.5412 |

### Classification Metrics
- **Risk Classifier (Random Forest):** Achieved **99.71% Accuracy** mapping profile data to risk levels.

**Analysis:** Gradient Boosting and XGBoost successfully mapped the non-linear realistic logic and the newly introduced NCB compounding discounts, achieving an outstanding $R^2$ score of ~0.93 on core metrics.

👉 **Tools Used:** `scikit-learn.metrics` (`r2_score`, `mean_absolute_error`)

---

---

## 🔟 Model Tuning
**Process:** Rather than relying on default parameter guesses, `GridSearchCV` combined with 5-Fold Cross-Validation (`cv=5`) to optimize parameter geometries programmatically.

- **Random Forest:** Tuned across sets mapping `n_estimators` (100, 200) and tree `max_depth` (None, 10, 20).
- **Gradient Boosting:** Tuned identifying optimal `learning_rate` constraints and tree ensemble sizing.

👉 **Tools Used:** `scikit-learn.model_selection` (`GridSearchCV`, `KFold`)

---

## 1️⃣1️⃣ Model Deployment
**Process:** Transporting the static localized PKL files (.pkl pipelines, objects, encoders) into a dynamic, user-facing, real-world utility architecture.

- **Platform:** Formatted as a real-time reactive Web Application.
- **Key Real-World App Features:** 
  1. Dynamic localized Feature Impacts (XAI algorithms) detailing exactly *what* is contributing to a user's pricing.
  2. "AI Savings Advisor" - localized continuous inference calculating exact counter-factual reductions based on lifestyle choices and **No Claim Bonus** progression.
  3. **Statistical EDA Dashboard** - A dedicated visualization tab rendering pre-computed actuarial graphs (Distributions, Heatmaps, Box-plots) for each insurance category.

👉 **Tools Used:** `Streamlit`, `joblib`, `matplotlib`

---

## 1️⃣2️⃣ Monitoring & Maintenance
**Process:** Preparing the system topology for long-term data ingestion.

- **Logging:** An established connection strictly pushing all user sessions, inputted dynamic parameters, inference timestamps, and resulting prediction arrays into a local NoSQL database via `collections.insert_one()`.
- **Reasoning:** Storing incoming production data allows for data drift calculations, which signal when to retrain the underlying models (e.g., as worldwide medical inflation costs change over the years).

👉 **Tools Used:**  `pymongo`, `MongoDB`

## 📈 Comprehensive Exploratory Data Analysis (EDA)

This section was automatically generated by iterating across all insurance datasets to find key analytical relationships. Visual plots mapping detailed correlations, Box-Whisker category spreads, and Target Distributions have been rendered locally in the `eda_outputs` repository directory.
### Business Insurance EDA Summary
- **Total Records:** 10,000
- **Target Variable:** `Annual_Premium` (Mean: 105470.38)
- **Strongest Positive Correlation:** `Coverage_Amount` (0.74)
- **Strongest Negative Correlation:** `No_Claim_Years` (-0.08) (Note: `No_Claim_Years` usually has a strong negative correlation due to the compounded 2% discount strategy)

### Health Insurance EDA Summary
- **Total Records:** 10,000
- **Target Variable:** `Annual_Premium` (Mean: 26171.64)
- **Strongest Positive Correlation:** `Coverage_Amount` (0.77)
- **Strongest Negative Correlation:** `No_Claim_Years` (-0.06) (Note: `No_Claim_Years` usually has a strong negative correlation due to the compounded 2% discount strategy)

### Life Insurance EDA Summary
- **Total Records:** 1,000
- **Target Variable:** `Annual_Premium` (Mean: 26654.41)
- **Strongest Positive Correlation:** `Coverage_Amount` (0.79)
- **Strongest Negative Correlation:** `No_Claim_Years` (-0.12) (Note: `No_Claim_Years` usually has a strong negative correlation due to the compounded 2% discount strategy)

### Motor Insurance EDA Summary
- **Total Records:** 10,000
- **Target Variable:** `Annual_Premium` (Mean: 15237.62)
- **Strongest Positive Correlation:** `Coverage_Amount` (0.75)
- **Strongest Negative Correlation:** `No_Claim_Years` (-0.07) (Note: `No_Claim_Years` usually has a strong negative correlation due to the compounded 2% discount strategy)

### Property Insurance EDA Summary
- **Total Records:** 10,000
- **Target Variable:** `Annual_Premium` (Mean: 29481.26)
- **Strongest Positive Correlation:** `Property_Value` (0.79)
- **Strongest Negative Correlation:** `No_Claim_Years` (-0.05) (Note: `No_Claim_Years` usually has a strong negative correlation due to the compounded 2% discount strategy)

### Specialty Insurance EDA Summary
- **Total Records:** 10,000
- **Target Variable:** `Annual_Premium` (Mean: 24461.07)
- **Strongest Positive Correlation:** `Coverage_Amount` (0.73)
- **Strongest Negative Correlation:** `No_Claim_Years` (-0.07) (Note: `No_Claim_Years` usually has a strong negative correlation due to the compounded 2% discount strategy)

### Travel Insurance EDA Summary
- **Total Records:** 10,000
- **Target Variable:** `Premium` (Mean: 4974.57)
- **Strongest Positive Correlation:** `Coverage_Amount` (0.80)
- **Strongest Negative Correlation:** `No_Claim_Years` (-0.05) (Note: `No_Claim_Years` usually has a strong negative correlation due to the compounded 2% discount strategy)

### Claim Insurance EDA Summary
- **Total Records:** 50,000
- **Target Variable:** `claim` (Mean: 5548.42)
- **Strongest Positive Correlation:** `age` (0.62)
- **Strongest Negative Correlation:** `No_Claim_Years` (-0.07) (Note: `No_Claim_Years` usually has a strong negative correlation due to the compounded 2% discount strategy)


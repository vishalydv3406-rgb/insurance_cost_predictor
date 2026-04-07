import pandas as pd
import numpy as np

# Increased the number of tuples to 50,000 as requested
n = 50000

# 1. Age
age = np.random.randint(18, 65, n)
gender = np.random.choice(["male", "female"], n)

# 2. BMI: Normal distribution centered at 28
bmi = np.random.normal(loc=28.0, scale=6.0, size=n)
bmi = np.clip(bmi, 15.0, 55.0).round(1)

# 3. Blood Pressure: Normal distribution centered at 120
bloodpressure = np.random.normal(loc=120.0, scale=15.0, size=n)
bloodpressure = np.clip(bloodpressure, 80, 180).astype(int)

# 4. Children: Poisson distribution for realistic family sizes
children = np.random.poisson(lam=1.2, size=n)
children = np.clip(children, 0, 5)

# 5. Smoker: ~20% probability global average
smoker = np.random.choice(["Yes", "No"], p=[0.2, 0.8], size=n)

# 6. Diabetic: Dependent on Age and BMI
# Base prob 5%, goes up by 0.5% per year over 40, and 1% per BMI point over 30
diabetic_prob = 0.05 + np.maximum(0, (age - 40) * 0.005) + np.maximum(0, (bmi - 30) * 0.01)
diabetic_prob = np.clip(diabetic_prob, 0.01, 0.85)
random_draws = np.random.uniform(0, 1, n)
diabetic = np.where(random_draws < diabetic_prob, "Yes", "No")

region = np.random.choice(["northeast", "northwest", "southeast", "southwest"], n)

data = pd.DataFrame({
    "Id": range(1, n+1),
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "bloodpressure": bloodpressure,
    "diabetic": diabetic,
    "children": children,
    "smoker": smoker,
    "region": region,
})

# 7. Generate claim based on realistic actuarial logic
base_cost = 2000
age_penalty = ((data["age"] - 18) ** 1.3) * 30
bmi_penalty = np.maximum(data["bmi"] - 25, 0) * 120
children_addition = data["children"] * 400

# Applying Multipliers
smoker_mult = np.where(data["smoker"] == "Yes", 1.8, 1.0) # Smokers pay 80% more
diabetic_mult = np.where(data["diabetic"] == "Yes", 1.3, 1.0) # Diabetics pay 30% more

pure_premium = (base_cost + age_penalty + bmi_penalty + children_addition) * smoker_mult * diabetic_mult

# Heteroscedastic Noise (Varies proportionally to the premium)
noise = np.random.normal(0, pure_premium * 0.1)
data["claim"] = (pure_premium + noise).round(2)

# 8. No Claim Bonus (NCB)
data["No_Claim_Years"] = np.random.randint(0, 6, size=n)
discount_factor = 0.98 ** data["No_Claim_Years"]
data["claim"] = (data["claim"] * discount_factor).round(2)

data["claim"] = np.maximum(data["claim"], 1000) # Floor


# FDS Concept: Inject Missing Values (approximately 5% missing per selected column)
def inject_nans(df, col_name, missing_frac=0.05):
    n_missing = int(len(df) * missing_frac)
    missing_indices = np.random.choice(df.index, n_missing, replace=False)
    # Important: Convert column to float if it's integer so it can hold np.nan
    if pd.api.types.is_numeric_dtype(df[col_name]):
         df[col_name] = df[col_name].astype(float)
    df.loc[missing_indices, col_name] = np.nan

inject_nans(data, "bmi")
inject_nans(data, "bloodpressure")
inject_nans(data, "children")

# Save dataset directly to insurance.csv so the app uses the new larger dataset
data.to_csv("insurance.csv", index=False)

print(f"Successfully generated {n} rows of data and saved to insurance.csv!")
print(data.head())

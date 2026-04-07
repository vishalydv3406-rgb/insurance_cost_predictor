import pandas as pd
import numpy as np
import glob
from pathlib import Path

BASE_DIR = Path(__file__).parent

def apply_ncb_to_file(filepath):
    df = pd.read_csv(filepath)
    filename = Path(filepath).name
    
    if 'No_Claim_Years' in df.columns:
        print(f"Skipping {filename}, already has No_Claim_Years")
        return
        
    print(f"Applying NCB to {filename}...")
    
    # Generate random No_Claim_Years (0 to 5)
    np.random.seed(42)  # For reproducibility
    df['No_Claim_Years'] = np.random.randint(0, 6, size=len(df))
    
    # Determine the target column
    target_col = None
    if filename == "insurance.csv":
        target_col = "claim"
    else:
        if 'Premium_Amount' in df.columns:
            target_col = 'Premium_Amount'
        elif 'Premium' in df.columns and 'Annual_Premium' not in df.columns:
            target_col = 'Premium'
        elif 'Annual_Premium' in df.columns:
            target_col = 'Annual_Premium'
            
    if target_col and target_col in df.columns:
        # 2% compounding discount per year
        discount_factor = 0.98 ** df['No_Claim_Years']
        df[target_col] = (df[target_col] * discount_factor).round(2)
        df.to_csv(filepath, index=False)
        print(f"Successfully applied NCB logic to {filename} on column {target_col}")
    else:
        print(f"Target column not found in {filename}, could not apply discount.")

if __name__ == "__main__":
    csv_files = glob.glob(str(BASE_DIR / '*_insurance_*.csv'))
    csv_files.append(str(BASE_DIR / 'insurance.csv'))
    for f in csv_files:
        apply_ncb_to_file(f)

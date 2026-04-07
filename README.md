# 🏥 Insurance Payment Predictor

An AI-powered Streamlit app that predicts health insurance costs based on customer profile. Built with Python, scikit-learn, and Streamlit.

## Features

- **Cost Prediction** – Estimate insurance claims using age, BMI, blood pressure, smoking, diabetic status, and more
- **Data Explorer** – Interactive visualizations of the insurance dataset
- **Feature Importance** – Understand which factors most influence predictions
- **Modern UI** – Dark theme with teal/amber accents

## Setup

1. **Clone or download** this project.

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # or: source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files** are present in the project root:
   - `scaler.pkl`
   - `label_encoder_gender.pkl`
   - `label_encoder_diabetic.pkl`
   - `label_encoder_smoker.pkl`
   - `best_model.pkl`
   - `insurance.csv`

## Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project Structure

```
├── app.py              # Streamlit application
├── insurance.csv       # Dataset
├── best_model.pkl      # Trained Random Forest model
├── scaler.pkl          # Feature scaler
├── label_encoder_*.pkl # Label encoders
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Disclaimer

This is an educational project. Predictions are estimates and should not be used for actual insurance or financial decisions.

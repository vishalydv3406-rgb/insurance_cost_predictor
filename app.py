import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import pymongo
from datetime import datetime
import json
import glob
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split

# --- Configuration ---
st.set_page_config(
    page_title="Insurance AI Pro+",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    :root {
        --background: #09090b;
        --foreground: #fafafa;
        --card: #09090b;
        --card-foreground: #fafafa;
        --primary: #fafafa;
        --primary-foreground: #18181b;
        --secondary: #27272a;
        --secondary-foreground: #fafafa;
        --muted: #27272a;
        --muted-foreground: #a1a1aa;
        --border: #27272a;
        --radius: 0.5rem;
    }
    
    .stApp {
        background-color: var(--background);
        color: var(--foreground);
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        text-align: center;
        color: var(--foreground);
        margin-bottom: 2rem;
    }
    .card {
        background-color: var(--card);
        color: var(--card-foreground);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border);
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stButton>button {
        background-color: var(--primary);
        color: var(--primary-foreground);
        border: none;
        padding: 0.5rem 1rem;
        border-radius: var(--radius);
        font-weight: 500;
        transition: background-color 0.2s ease;
        width: 100%;
        height: 2.5rem;
    }
    .stButton>button:hover {
        background-color: #e4e4e7;
        color: var(--primary-foreground);
    }
    button[kind="primary"] {
        background-color: var(--primary) !important;
        color: var(--primary-foreground) !important;
    }
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# --- Dynamic Data Loading ---
BASE_DIR = Path(__file__).parent

@st.cache_resource
def init_connection():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.server_info()
        db = client["insurance_db"]
        return db["predictions_dynamic"]
    except Exception as e:
        return None

collection = init_connection()

@st.cache_data
def get_available_insurances():
    configs = glob.glob(str(BASE_DIR / "*_config.json"))
    insurances = [Path(c).name.split("_config")[0].capitalize() for c in configs]
    return sorted(insurances)

@st.cache_data
def get_dataset_map():
    csvs = glob.glob(str(BASE_DIR / '*_insurance_*.csv'))
    csvs.append(str(BASE_DIR / 'insurance.csv'))
    mapping = {}
    for c in csvs:
        name = Path(c).name
        if name == "insurance.csv":
            mapping["Claim"] = c
        else:
            t = name.split('_insurance')[0].capitalize()
            mapping[t] = c
    return mapping

@st.cache_resource
def load_dynamic_artifacts(insurance_type):
    name_lower = insurance_type.lower()
    model_path = BASE_DIR / f"{name_lower}_models.pkl"
    config_path = BASE_DIR / f"{name_lower}_config.json"
    
    if not model_path.exists() or not config_path.exists():
        return None, None
        
    pipelines = joblib.load(model_path)
    with open(config_path, "r") as f:
        config = json.load(f)
        
    
    return pipelines, config

@st.cache_data
def get_model_performance(insurance_type):
    """Calculates R2 and MAE for the models of the selected insurance type."""
    name_lower = insurance_type.lower()
    dataset_map = get_dataset_map()
    current_csv = dataset_map.get(insurance_type)
    
    if not current_csv or not Path(current_csv).exists():
        return None
        
    df = pd.read_csv(current_csv)
    
    # Load Models and Config
    pipelines, config = load_dynamic_artifacts(insurance_type)
    if not pipelines:
        return None
        
    # Prepare Data (logic from train_all_models.py)
    if insurance_type == "Claim":
        target_col = "claim"
        drop_cols = ["Id"]
    else:
        target_col = config.get("target_col", "Premium_Amount")
        drop_cols = ['Policy_ID', 'Claim_Status']

    y = df[target_col]
    X = df.drop(columns=[col for col in drop_cols + [target_col] if col in df.columns])
    
    # Split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    performance = []
    sample_preds = None
    
    for name, pipeline in pipelines.items():
        try:
            y_pred = pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            performance.append({
                "Model": name,
                "R2_Score": r2,
                "MAE": mae
            })
            
            # Use GB as sample if available, else first model
            if name == "Gradient Boosting" or sample_preds is None:
                sample_preds = pd.DataFrame({
                    "Actual": y_test[:10].values,
                    "Predicted": [round(v, 2) for v in y_pred[:10]]
                })
        except:
            continue
            
    # Also evaluate risk classifier if insurance is Claim
    risk_acc = None
    if insurance_type == "Claim":
        try:
            risk_clf_path = BASE_DIR / "risk_classifier.pkl"
            if risk_clf_path.exists():
                # We need the main insurance.csv preprocessing for the main classifier
                # This is tricky because it uses pre-poly features.
                # For now, let's just stick to the regression metrics.
                pass
        except:
            pass
            
    return {
        "metrics": pd.DataFrame(performance),
        "sample": sample_preds
    }

# --- Main Layout ---
st.markdown('<div class="main-header">🛡️ Insurance AI Pro+</div>', unsafe_allow_html=True)

available_insurances = get_available_insurances()
if not available_insurances:
    st.error("No trained models found! Please run the training script first.")
    st.stop()

# Selection Header
selected_insurance = st.selectbox(
    "Select the Type of Insurance to Analyze:",
    options=available_insurances,
    index=0
)

st.markdown("---")

pipelines, config = load_dynamic_artifacts(selected_insurance)

dataset_map = get_dataset_map()
current_csv = dataset_map.get(selected_insurance)
df = pd.read_csv(current_csv) if current_csv else None

if pipelines is None:
    st.error(f"Failed to load artifacts for {selected_insurance} Insurance.")
    st.stop()

col1, col2 = st.columns([1, 2.5], gap="large")

with col1:
    st.markdown(f"### 📋 {selected_insurance} Details")
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Dynamic Input Generation
        input_data = {}
        target_name = config["target_col"]
        
        # Random Generator
        if st.button("🎲 Generate Random Details", use_container_width=True):
            for num_feat in config["features"]["numerical"]:
                fname = num_feat["name"]
                st.session_state[f"rand_{fname}"] = float(np.round(np.random.uniform(num_feat["min"], num_feat["max"]), 1))
            for cat_feat, cat_opts in config["features"]["categorical"].items():
                st.session_state[f"rand_{cat_feat}"] = np.random.choice(cat_opts)
                
        # Render Numerical Inputs
        for num_feat in config["features"]["numerical"]:
            fname = num_feat["name"]
            
            # Determine step and type
            if "age" in fname.lower() or "children" in fname.lower() or "size" in fname.lower() or "days" in fname.lower():
                default_val = int(st.session_state.get(f"rand_{fname}", num_feat["mean"]))
                input_data[fname] = st.slider(
                    fname.replace("_", " "), 
                    int(num_feat["min"]), 
                    int(num_feat["max"]), 
                    default_val
                )
            else:
                default_val = float(st.session_state.get(f"rand_{fname}", num_feat["mean"]))
                input_data[fname] = st.number_input(
                    fname.replace("_", " "), 
                    min_value=float(num_feat["min"]), 
                    max_value=float(num_feat["max"]), 
                    value=default_val,
                    step=1.0
                )
                
        # Render Categorical Inputs
        for cat_feat, cat_opts in config["features"]["categorical"].items():
            default_val = st.session_state.get(f"rand_{cat_feat}", cat_opts[0])
            idx = cat_opts.index(default_val) if default_val in cat_opts else 0
            input_data[cat_feat] = st.selectbox(
                cat_feat.replace("_", " "), 
                options=cat_opts, 
                index=idx
            )
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    predict_btn = st.button("🔮 Estimate Cost", type="primary", use_container_width=True)

with col2:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Prediction Analysis", "👁️ Dynamic Feature Impacts", "🤖 Model Comparison", "🎒 My Portfolio", "📈 Data Explorer", "📈 Statistical EDA", "🏆 Model Accuracy"])

    with tab1:
        if predict_btn or "last_prediction" in st.session_state:
            try:
                # If 'predict_btn' was clicked, we recalculate and override the last prediction
                if predict_btn:
                    params = pd.DataFrame([input_data])
                    
                    # Predict across all logged models
                    all_preds = {}
                    for model_name, pipeline in pipelines.items():
                        all_preds[model_name] = pipeline.predict(params)[0]
                    
                    primary_pred = all_preds.get("Gradient Boosting", list(all_preds.values())[0])
                    
                    st.session_state.last_prediction = primary_pred
                    st.session_state.last_params = params.copy()
                    st.session_state.last_all_preds = all_preds
                    st.session_state.last_type = selected_insurance
                    
                    # Save to MongoDB
                    if collection is not None:
                        record = input_data.copy()
                        record["insurance_type"] = selected_insurance
                        record["predictions"] = all_preds
                        record["timestamp"] = datetime.now()
                        collection.insert_one(record)
                        st.success("✅ Prediction saved to database!")
                
                # Render Results based on the current session state
                pred_type = st.session_state.last_type
                pred_val = st.session_state.last_prediction
                all_preds = st.session_state.last_all_preds
                
                st.markdown(f"### 🏥 {pred_type} Cost Estimation")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                     st.metric(f"Estimated Annual Cost", f"₹{pred_val:,.0f}", delta_color="inverse")
                with res_col2:
                    st.metric("Monthly Equivalent", f"₹{pred_val/12:,.0f}")

                # --- Dynamic Cost Reduction Savings ---
                st.markdown("---")
                st.markdown("### 💡 AI Savings Advisor")
                
                savings_opportunities = []
                recent_df = st.session_state.last_params.copy()
                
                # Test categorical variations for savings
                if "categorical" in config["features"]:
                    for cat_feat, cat_opts in config["features"]["categorical"].items():
                        current_val = recent_df[cat_feat].iloc[0]
                        for opt in cat_opts:
                            if opt != current_val:
                                temp_df = recent_df.copy()
                                temp_df[cat_feat] = opt
                                # Use Gradient Boosting for the test estimation
                                test_pred = pipelines["Gradient Boosting"].predict(temp_df)[0]
                                savings = pred_val - test_pred
                                if savings > 5: # Threshold to ignore tiny float artifacts
                                    savings_opportunities.append({
                                        "feature": cat_feat,
                                        "from_val": current_val,
                                        "to_val": opt,
                                        "savings": savings
                                    })
                                    
                if savings_opportunities:
                    # Sort by highest saving
                    savings_opportunities = sorted(savings_opportunities, key=lambda x: x["savings"], reverse=True)
                    st.success("Based on our AI analysis, here are actionable ways you can lower your premium right now:")
                    for opp in savings_opportunities:
                        st.markdown(f"- **Optimize {opp['feature'].replace('_', ' ')}**: Changing from '{opp['from_val']}' to '{opp['to_val']}' could save you **₹{opp['savings']:,.0f}** annually.")
                else:
                    st.info("Your policy is highly optimized based on current categorical selections. No further direct savings found without altering key coverages.")

                # --- Maximum Claim Criteria (Static Domain Knowledge) ---
                max_claim_rules = {
                    "Business": "Maintain detailed historical inventory records, implement strong security protocols, and file a formal report within 24 hours of any incident to avoid depreciation penalties.",
                    "Claim": "Submit all original bills and extensive photo documentation. Secure pre-approval for any elective procedures or major expenditures before proceeding.",
                    "Health": "Always seek care within the approved 'In-Network' hospital list and obtain explicit pre-authorization for non-emergency surgeries. Retain every discharge summary.",
                    "Life": "Ensure all medical history was 100% truthfully disclosed at application time, assign updated clear beneficiaries, and ensure zero lapses in premium payment.",
                    "Motor": "Never move the vehicle post-accident until documented. Immediately file a police FIR for severe collisions or theft incidents to substantiate the insurance claim fully.",
                    "Property": "Maintain an annual video walkthrough inventory of the property. Take immediate mitigative action (e.g. turning off main water during a leak) before the surveyor arrives.",
                    "Specialty": "Maintain professional, certified periodic appraisals of the items. Keep proof-of-ownership and authenticity certificates in a remote safe deposit box.",
                    "Travel": "Obtain verifiable local police/medical reports immediately upon incident abroad. Preserve all original receipts for enforced delays, such as extra hotel/flight expenses."
                }
                
                st.markdown("### 🏆 Maximizing Your Claim Value")
                rule = max_claim_rules.get(pred_type, "Always keep robust documentation and communicate with the insurer immediately after any qualifying event.")
                st.info(f"**How to ensure maximum payout for {pred_type} Insurance:** {rule}")

                # --- No Claim Bonus Advisor ---
                if "No_Claim_Years" in input_data:
                    ncb_years = input_data["No_Claim_Years"]
                    if ncb_years < 5:
                        st.warning(f"💡 **NCB Opportunity**: You currently have {ncb_years} No Claim Years. By staying claim-free, you will earn an additional **2% compounding discount** per year, up to a maximum of 5 years (currently you're {(5-ncb_years)} years away from max bonus).")
                    else:
                        st.success("🌟 **Maximum NCB Reached!** You are enjoying the full 5-year No Claim Bonus discount.")

                # Portfolio Addition System
                st.markdown("---")
                if st.button("➕ Add to My Portfolio", type="secondary", use_container_width=True):
                    # Check if already added to prevent duplicate spam, but allow multiple lines
                    st.session_state.portfolio.append({
                        "id": len(st.session_state.portfolio) + 1,
                        "insurance_type": pred_type,
                        "cost": pred_val,
                        "details": st.session_state.last_params.to_dict('records')[0]
                    })
                    st.success(f"{pred_type} Insurance saved to your portfolio! Head over to the 'My Portfolio' tab to view overall analysis.")

                st.markdown("---")
                st.success(f"**Gradient Boosting is the recommended model** for {pred_type}. It provides an ensemble estimation correcting the errors of baseline decision trees.")
                
                csv_data = "Model,Predicted_Cost\n"
                for m_name, m_val in all_preds.items():
                    csv_data += f"{m_name},{m_val:.2f}\n"
                st.download_button("📥 Download Cost Report", data=csv_data, file_name=f"{pred_type}_quote.csv", mime="text/csv")


            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
        else:
            st.info("👈 Configure the details on the left and click 'Estimate Cost' to begin.")

    with tab2:
        st.markdown("## 👁️ Local Feature Impacts")
        st.info("What is driving your specific premium? We modify your exact inputs to a generic baseline to isolate factors.")
        
        if 'last_params' not in st.session_state or st.session_state.get('last_type') != selected_insurance:
            st.warning("Please submit a profile in the first tab to see insights.")
        else:
            recent_df = st.session_state.last_params.copy()
            base_pred = pipelines["Gradient Boosting"].predict(recent_df)[0]
            
            impacts = {}
            for col in recent_df.columns:
                temp_df = recent_df.copy()
                
                # Set numeric to their mean, categorical to their first element
                is_num = False
                for feat in config["features"]["numerical"]:
                    if feat["name"] == col:
                        temp_df[col] = feat["mean"]
                        is_num = True
                        break
                if not is_num:
                    temp_df[col] = config["features"]["categorical"][col][0]
                
                mod_pred = pipelines["Gradient Boosting"].predict(temp_df)[0]
                impacts[col] = base_pred - mod_pred
                
            sorted_impacts = dict(sorted(impacts.items(), key=lambda item: abs(item[1]), reverse=True))
            
            impact_df = pd.DataFrame({
                "Factor": list(sorted_impacts.keys()),
                "Impact on Premium (₹)": list(sorted_impacts.values())
            })
            
            fig_impact = px.bar(
                impact_df, 
                x="Impact on Premium (₹)", 
                y="Factor", 
                orientation='h',
                color="Impact on Premium (₹)",
                color_continuous_scale=px.colors.diverging.RdYlGn[::-1],
                title="How Your Attributes Shift Your Price vs Average Baseline"
            )
            fig_impact.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_impact, use_container_width=True)

    with tab3:
        st.markdown("### 🤖 Model Comparison")
        if 'last_all_preds' in st.session_state and st.session_state.get('last_type') == selected_insurance:
            all_preds = st.session_state.last_all_preds
            st.markdown(f"Comparison of AI algorithms for **{st.session_state.last_type}** prediction:")
            
            cols = st.columns(len(all_preds))
            for i, (m_name, cost) in enumerate(all_preds.items()):
                with cols[i]:
                    st.markdown(f"**{m_name}**")
                    st.markdown(f"#### ₹{cost:,.0f}")
        else:
            st.warning("Please run a prediction first.")

    with tab4:
        st.markdown("### 🎒 My Insurance Portfolio")
        if not st.session_state.portfolio:
            st.info("Your portfolio is empty. Add predictions from the Analysis tab to build your profile.")
        else:
            portfolio_df = pd.DataFrame([{ 
                "Portfolio ID": p["id"],
                "Insurance Type": p["insurance_type"], 
                "Estimated Annual Cost": p["cost"],
                "Monthly Premium": p["cost"]/12
            } for p in st.session_state.portfolio])
            
            total_annual = portfolio_df["Estimated Annual Cost"].sum()
            total_monthly = total_annual / 12
            
            st.markdown("#### Portfolio Summary")
            m1, m2 = st.columns(2)
            m1.metric("Total Annual Insurance Cost", f"₹{total_annual:,.0f}")
            m2.metric("Total Monthly Premium", f"₹{total_monthly:,.0f}")
            
            st.dataframe(portfolio_df.style.format({"Estimated Annual Cost": "₹{:,.0f}", "Monthly Premium": "₹{:,.0f}"}), use_container_width=True)
            
            if len(st.session_state.portfolio) > 1:
                st.markdown("#### Cost Distribution")
                # Group by insurance type in case there are duplicates
                grouped_df = portfolio_df.groupby('Insurance Type')['Estimated Annual Cost'].sum().reset_index()
                fig = px.pie(grouped_df, names='Insurance Type', values='Estimated Annual Cost', title="Insurance Spend by Category", hole=0.3)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Overall Insight Analysis")
                types_bought = portfolio_df['Insurance Type'].unique().tolist()
                
                insights = []
                if "Health" in types_bought and "Life" in types_bought:
                    insights.append("✅ **Strong Personal Protection**: You have secured both health and life coverage, establishing a very strong safety net for yourself and dependents.")
                elif "Health" in types_bought and "Life" not in types_bought:
                    insights.append("💡 **Recommendation**: You have Health coverage, but lack Life Insurance. Consider adding Life Insurance to protect your family's long-term financial stability.")
                
                if "Business" in types_bought and "Property" not in types_bought:
                    insights.append("🏢 **Business Risk**: You have business insurance, but no dedicated property coverage. If you own physical commercial space, strongly consider a Property policy.")
                
                if "Motor" in types_bought and "Health" not in types_bought:
                    insights.append("⚠️ **Priority Imbalance**: You have mathematically insured your vehicle but not your own health. Medical emergencies are statistically costlier than vehicle repairs. Consider Health Insurance.")
                
                if "Travel" in types_bought and "Health" not in types_bought:
                    insights.append("✈️ **Travel Risk**: You have travel insurance but no baseline health insurance. Note that domestic health issues will not be covered by travel policies.")
                
                if len(insights) > 0:
                    for i in insights:
                        st.info(i)
                else:
                    st.success("✅ Great start! You are building a diversified portfolio of insurance coverage.")
            else:
                st.write("Add more insurances to unlock comprehensive cross-policy insights!")

    with tab5:
        st.markdown(f"### 📈 {selected_insurance} Data Explorer")
        if df is not None:
            st.write("Explore the underlying historical dataset that powers our AI.")
            
            # Target Distribution
            st.subheader("Target Distribution")
            fig_hist = px.histogram(df, x=target_name, nbins=50, title=f"Distribution of {target_name.replace('_', ' ')}")
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Correlation Matrix
            st.subheader("Feature Correlation Matrix")
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                corr = num_df.corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Numerical Correlations")
                fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig_corr, use_container_width=True)
                
            # Interactive Scatter
            st.subheader("Custom Scatter Analysis")
            sc_x = st.selectbox("X-Axis", df.columns, index=0)
            sc_y = st.selectbox("Y-Axis", df.columns, index=min(1, len(df.columns)-1))
            
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            sc_col = st.selectbox("Color By (Optional)", ["None"] + cat_cols, index=0)
            
            if sc_col != "None":
                fig_sc = px.scatter(df, x=sc_x, y=sc_y, color=sc_col, opacity=0.7, title=f"{sc_y} vs {sc_x} (Colored by {sc_col})")
            else:
                fig_sc = px.scatter(df, x=sc_x, y=sc_y, opacity=0.7, title=f"{sc_y} vs {sc_x}")
                
            fig_sc.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
            st.plotly_chart(fig_sc, use_container_width=True)
            
            # Categorical Boxplots
            if cat_cols:
                st.subheader("Categorical Breakdown")
                box_cat = st.selectbox("Group By", cat_cols, index=0)
                fig_box = px.box(df, x=box_cat, y=target_name, color=box_cat, title=f"{target_name} grouped by {box_cat}")
                fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Could not load the dataset for analysis.")

    with tab6:
        st.markdown(f"### 🧪 {selected_insurance} Statistical EDA")
        st.write("Visualized insights from the comprehensive automated EDA pipeline.")
        
        type_dir = BASE_DIR / "eda_outputs" / selected_insurance
        if type_dir.exists():
            images = glob.glob(str(type_dir / "*.png"))
            if images:
                # Show in a grid
                cols = st.columns(2)
                for i, img_path in enumerate(images):
                    with cols[i % 2]:
                        st.image(img_path, use_container_width=True, caption=Path(img_path).stem.replace("_", " ").title())
            else:
                st.info("No static graphs found for this insurance type.")
        else:
            st.warning("EDA images directory not found. Please run the EDA generation script.")

    with tab7:
        st.markdown(f"### 🏆 {selected_insurance} Model Performance Summary")
        st.info("Performance metrics evaluated on the test dataset (20% holdout).")
        
        with st.spinner("Calculating performance metrics..."):
            perf_data = get_model_performance(selected_insurance)
        
        if perf_data and not perf_data["metrics"].empty:
            m_metrics = perf_data["metrics"]
            m_sample = perf_data["sample"]
            
            # Metics row
            best_model_row = m_metrics.loc[m_metrics['R2_Score'].idxmax()]
            
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Best Model", best_model_row["Model"])
            with mc2:
                st.metric("Max R² Score", f"{best_model_row['R2_Score']:.4f}")
            with mc3:
                st.metric("Min Avg Error (MAE)", f"₹{best_model_row['MAE']:,.2f}")
            
            # Charts
            st.markdown("---")
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### $R^2$ Score Comparison")
                fig_r2 = px.bar(m_metrics, x="Model", y="R2_Score", color="Model", 
                                title=f"Model Accuracy ($R^2$)",
                                color_discrete_sequence=px.colors.qualitative.Prism)
                fig_r2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig_r2, use_container_width=True)
                
            with c2:
                st.markdown("#### Mean Absolute Error (MAE)")
                fig_mae = px.bar(m_metrics, x="Model", y="MAE", color="Model", 
                                 title="Average Prediction Error (Lower is better)",
                                 color_discrete_sequence=px.colors.qualitative.Safe)
                fig_mae.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig_mae, use_container_width=True)
                
            # Sample Predictions
            st.markdown("---")
            st.markdown("#### 🎯 Sample Predictions (Actual vs Predicted)")
            st.table(m_sample)
            
            # Final Accuracy Note
            acc_note = f"The **{best_model_row['Model']}** model shows the highest reliability for {selected_insurance} insurance with an R² of {best_model_row['R2_Score']:.4f}, meaning it explains {best_model_row['R2_Score']*100:.1f}% of the pricing variance."
            st.success(acc_note)
            
        else:
            st.warning("Performance metrics could not be calculated for this insurance type.")

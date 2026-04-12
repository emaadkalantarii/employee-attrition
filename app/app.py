# ============================================================
#  Employee Attrition Predictor — Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import os

# ── Page configuration ───────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f3c88;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #f8d7da;
        border-left: 5px solid #e74c3c;
        padding: 1rem;
        border-radius: 4px;
        font-size: 1.1rem;
        color: #7b1a1a !important;
    }
    .risk-high strong {
        color: #7b1a1a !important;
    }
    .risk-low {
        background-color: #d4edda;
        border-left: 5px solid #27ae60;
        padding: 1rem;
        border-radius: 4px;
        font-size: 1.1rem;
        color: #155724 !important;
    }
    .risk-low strong {
        color: #155724 !important;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models and artifacts ────────────────────────────────
@st.cache_resource
def load_artifacts():
    base       = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, '..', 'models')
    data_dir   = os.path.join(base, '..', 'data')

    with open(os.path.join(models_dir, 'best_model.pkl'),        'rb') as f:
        best_model = pickle.load(f)
    with open(os.path.join(models_dir, 'all_models.pkl'),        'rb') as f:
        all_models = pickle.load(f)
    with open(os.path.join(models_dir, 'scaler.pkl'),            'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, 'imputer.pkl'),           'rb') as f:
        imputer = pickle.load(f)
    with open(os.path.join(models_dir, 'selected_features.pkl'), 'rb') as f:
        selected_features = pickle.load(f)
    with open(os.path.join(models_dir, 'shap_explainer.pkl'),    'rb') as f:
        explainer = pickle.load(f)

    X_test  = pd.read_csv(os.path.join(data_dir, 'X_test_final.csv'))
    y_test  = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze()
    results = pd.read_csv(os.path.join(data_dir, 'model_results.csv'), index_col=0)

    return (best_model, all_models, scaler, imputer,
            selected_features, explainer, X_test, y_test, results)

(best_model, all_models, scaler, imputer,
 selected_features, explainer, X_test, y_test, results) = load_artifacts()


# ── Helper: build feature vector from user inputs ────────────
def build_feature_vector(inputs: dict, selected_features: list) -> pd.DataFrame:
    dept_sales     = 1 if inputs['Department'] == 'Sales' else 0

    role_map = {
        'Sales Executive': 0, 'Research Scientist': 0,
        'Laboratory Technician': 1, 'Manufacturing Director': 0,
        'Healthcare Representative': 0, 'Manager': 0,
        'Sales Representative': 0, 'Research Director': 0,
        'Human Resources': 0
    }
    role_lab_tech      = role_map.get(inputs['JobRole'], 0)
    travel_freq        = 1 if inputs['BusinessTravel'] == 'Travel_Frequently' else 0
    marital_single     = 1 if inputs['MaritalStatus'] == 'Single' else 0
    overtime           = 1 if inputs['OverTime'] == 'Yes' else 0

    age                = inputs['Age']
    monthly_income     = inputs['MonthlyIncome']
    distance           = inputs['DistanceFromHome']
    years_company      = inputs['YearsAtCompany']
    total_working      = inputs['TotalWorkingYears']
    years_role         = inputs['YearsInCurrentRole']
    years_manager      = inputs['YearsWithCurrManager']
    years_promo        = inputs['YearsSinceLastPromotion']
    num_companies      = inputs['NumCompaniesWorked']
    job_level          = inputs['JobLevel']
    stock_option       = inputs['StockOptionLevel']
    job_satisfaction   = inputs['JobSatisfaction']
    env_satisfaction   = inputs['EnvironmentSatisfaction']
    rel_satisfaction   = inputs['RelationshipSatisfaction']
    work_life          = inputs['WorkLifeBalance']
    job_involvement    = inputs['JobInvolvement']
    daily_rate         = inputs['DailyRate']
    monthly_rate       = inputs['MonthlyRate']
    hourly_rate        = inputs['HourlyRate']
    training_times     = inputs['TrainingTimesLastYear']
    pct_salary_hike    = inputs['PercentSalaryHike']

    # Engineered features — mirrors Phase 4
    income_per_exp      = monthly_income / (total_working + 1)
    years_without_promo = years_company - years_promo
    satisfaction_score  = (job_satisfaction + env_satisfaction +
                            rel_satisfaction + work_life) / 4
    manager_loyalty     = years_manager / (years_company + 1)
    career_growth       = job_level / (total_working + 1)
    is_early_career     = int(years_company <= 2)
    is_overdue_promo    = int(years_promo >= 4)

    row = {
        'Age'                              : age,
        'DailyRate'                        : daily_rate,
        'DistanceFromHome'                 : distance,
        'EnvironmentSatisfaction'          : env_satisfaction,
        'HourlyRate'                       : hourly_rate,
        'JobInvolvement'                   : job_involvement,
        'JobLevel'                         : job_level,
        'JobSatisfaction'                  : job_satisfaction,
        'MonthlyIncome'                    : monthly_income,
        'MonthlyRate'                      : monthly_rate,
        'NumCompaniesWorked'               : num_companies,
        'OverTime'                         : overtime,
        'PercentSalaryHike'                : pct_salary_hike,
        'RelationshipSatisfaction'         : rel_satisfaction,
        'StockOptionLevel'                 : stock_option,
        'TotalWorkingYears'                : total_working,
        'TrainingTimesLastYear'            : training_times,
        'WorkLifeBalance'                  : work_life,
        'YearsAtCompany'                   : years_company,
        'YearsInCurrentRole'               : years_role,
        'YearsSinceLastPromotion'          : years_promo,
        'YearsWithCurrManager'             : years_manager,
        'Department_Sales'                 : dept_sales,
        'JobRole_Laboratory Technician'    : role_lab_tech,
        'MaritalStatus_Single'             : marital_single,
        'BusinessTravel_Travel_Frequently' : travel_freq,
        'SatisfactionScore'                : satisfaction_score,
        'IncomePerYearExp'                 : income_per_exp,
        'YearsWithoutPromotion'            : years_without_promo,
        'CareerGrowthRate'                 : career_growth,
        'ManagerLoyaltyRatio'              : manager_loyalty,
        'IsEarlyCareer'                    : is_early_career,
        'IsOverduePromotion'               : is_overdue_promo,
    }

    df = pd.DataFrame([row])
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[selected_features]
    return df


# ════════════════════════════════════════════════════════════
#  SIDEBAR — Employee Input Form
# ════════════════════════════════════════════════════════════
st.sidebar.markdown("## 👤 Employee Profile")
st.sidebar.markdown("Fill in the employee details to predict attrition risk.")
st.sidebar.markdown("---")

with st.sidebar:
    st.markdown("**📋 Personal Info**")
    age             = st.slider("Age", 18, 60, 35)
    marital_status  = st.selectbox("Marital Status",
                                   ["Single", "Married", "Divorced"])
    distance        = st.slider("Distance From Home (km)", 1, 30, 10)

    st.markdown("---")
    st.markdown("**💼 Job Details**")
    department      = st.selectbox("Department",
                                   ["Research & Development", "Sales",
                                    "Human Resources"])
    job_role        = st.selectbox("Job Role", [
                                   "Sales Executive", "Research Scientist",
                                   "Laboratory Technician",
                                   "Manufacturing Director",
                                   "Healthcare Representative", "Manager",
                                   "Sales Representative", "Research Director",
                                   "Human Resources"])
    job_level       = st.selectbox("Job Level (1=Entry, 5=Executive)",
                                   [1, 2, 3, 4, 5])
    overtime        = st.selectbox("Works Overtime?", ["No", "Yes"])
    business_travel = st.selectbox("Business Travel",
                                   ["Non-Travel", "Travel_Rarely",
                                    "Travel_Frequently"])

    st.markdown("---")
    st.markdown("**💰 Compensation**")
    monthly_income  = st.slider("Monthly Income ($)", 1000, 20000, 5000, step=100)
    daily_rate      = st.slider("Daily Rate", 100, 1500, 700)
    hourly_rate     = st.slider("Hourly Rate", 30, 100, 65)
    monthly_rate    = st.slider("Monthly Rate", 2000, 27000, 14000, step=500)
    pct_hike        = st.slider("Percent Salary Hike (%)", 11, 25, 15)
    stock_option    = st.selectbox("Stock Option Level (0–3)", [0, 1, 2, 3])

    st.markdown("---")
    st.markdown("**📅 Experience & Tenure**")
    total_working   = st.slider("Total Working Years", 0, 40, 10)
    years_company   = st.slider("Years at Company", 0, 40, 5)
    years_role      = st.slider("Years in Current Role", 0, 18, 3)
    years_manager   = st.slider("Years with Current Manager", 0, 17, 3)
    years_promo     = st.slider("Years Since Last Promotion", 0, 15, 2)
    num_companies   = st.slider("Number of Companies Worked", 0, 9, 2)
    training_times  = st.slider("Training Times Last Year", 0, 6, 3)

    st.markdown("---")
    st.markdown("**😊 Satisfaction Scores (1=Low, 4=High)**")
    job_satisfaction = st.selectbox("Job Satisfaction",         [1, 2, 3, 4], index=2)
    env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
    rel_satisfaction = st.selectbox("Relationship Satisfaction",[1, 2, 3, 4], index=2)
    work_life        = st.selectbox("Work-Life Balance",        [1, 2, 3, 4], index=2)
    job_involvement  = st.selectbox("Job Involvement",          [1, 2, 3, 4], index=2)


# ════════════════════════════════════════════════════════════
#  MAIN PAGE
# ════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">👥 Employee Attrition Predictor</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict whether an employee is at risk of leaving '
            '— powered by XGBoost + SHAP interpretability</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🎯 Prediction",
    "📊 Model Performance",
    "📖 About"
])


# ════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════
with tab1:
    user_inputs = {
        'Age': age, 'MaritalStatus': marital_status,
        'DistanceFromHome': distance, 'Department': department,
        'JobRole': job_role, 'JobLevel': job_level,
        'OverTime': overtime, 'BusinessTravel': business_travel,
        'MonthlyIncome': monthly_income, 'DailyRate': daily_rate,
        'HourlyRate': hourly_rate, 'MonthlyRate': monthly_rate,
        'PercentSalaryHike': pct_hike, 'StockOptionLevel': stock_option,
        'TotalWorkingYears': total_working, 'YearsAtCompany': years_company,
        'YearsInCurrentRole': years_role, 'YearsWithCurrManager': years_manager,
        'YearsSinceLastPromotion': years_promo, 'NumCompaniesWorked': num_companies,
        'TrainingTimesLastYear': training_times,
        'JobSatisfaction': job_satisfaction,
        'EnvironmentSatisfaction': env_satisfaction,
        'RelationshipSatisfaction': rel_satisfaction,
        'WorkLifeBalance': work_life, 'JobInvolvement': job_involvement,
    }

    feature_vector = build_feature_vector(user_inputs, selected_features)

    prob     = best_model.predict_proba(feature_vector)[0][1]
    pred     = best_model.predict(feature_vector)[0]
    risk_pct = prob * 100

    col1, col2, col3 = st.columns([1.5, 1, 1])

    with col1:
        if pred == 1:
            st.markdown(f"""
            <div class="risk-high">
                🚨 <strong>HIGH ATTRITION RISK</strong><br><br>
                This employee has a <strong>{risk_pct:.1f}%</strong>
                probability of leaving.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                ✅ <strong>LOW ATTRITION RISK</strong><br><br>
                This employee has a <strong>{risk_pct:.1f}%</strong>
                probability of leaving.
            </div>""", unsafe_allow_html=True)

    with col2:
        st.metric("Attrition Probability", f"{risk_pct:.1f}%")
        st.metric("Prediction", "Will Leave 🚨" if pred == 1 else "Will Stay ✅")

    with col3:
        fig_gauge, ax = plt.subplots(figsize=(3, 1.2))
        ax.barh([0], [100], color='#e8e8e8', height=0.5)
        color = '#e74c3c' if prob > 0.5 else '#f39c12' if prob > 0.3 else '#27ae60'
        ax.barh([0], [risk_pct], color=color, height=0.5)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Risk %")
        ax.axvline(50, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title("Risk Gauge", fontsize=9)
        fig_gauge.tight_layout()
        st.pyplot(fig_gauge)
        plt.close()

    st.markdown("---")
    st.subheader("🔍 Why did the model make this prediction?")
    st.caption("SHAP values show which factors drove this specific prediction. "
               "Red = increases attrition risk  |  Blue = decreases attrition risk.")

    model_type  = type(best_model).__name__
    shap_single = explainer.shap_values(feature_vector)

    if model_type == 'RandomForestClassifier':
        shap_single_vals = shap_single[:, :, 1]
        base_val         = explainer.expected_value[1]
    else:
        shap_single_vals = shap_single
        base_val         = explainer.expected_value

    explanation = shap.Explanation(
        values        = shap_single_vals[0],
        base_values   = base_val,
        data          = feature_vector.iloc[0].values,
        feature_names = selected_features
    )

    fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(fig_shap)
    plt.close()

    st.markdown("---")
    with st.expander("📋 View full input feature vector"):
        st.dataframe(feature_vector.T.rename(columns={0: 'Value'}),
                     use_container_width=True)


# ════════════════════════════════════════════════════════════
#  TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Model Comparison")
    st.caption("Performance of all three trained models on the held-out test set.")

    # ── Results table — no background highlighting (avoids contrast issues) ──
    st.dataframe(
        results.style.format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("🔬 Global Feature Importance (SHAP)")

    img_col1, img_col2 = st.columns(2)
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data')

    with img_col1:
        shap_bar_path = os.path.join(base_path, 'shap_bar.png')
        if os.path.exists(shap_bar_path):
            st.image(shap_bar_path,
                     caption="Mean |SHAP| — Feature Importance Ranking",
                     use_container_width=True)

    with img_col2:
        shap_summary_path = os.path.join(base_path, 'shap_summary.png')
        if os.path.exists(shap_summary_path):
            st.image(shap_summary_path,
                     caption="SHAP Beeswarm — Direction & Magnitude",
                     use_container_width=True)

    st.markdown("---")
    st.subheader("📈 ROC Curve & Confusion Matrices")

    img_col3, img_col4 = st.columns(2)

    with img_col3:
        roc_path = os.path.join(base_path, 'roc_curves.png')
        if os.path.exists(roc_path):
            st.image(roc_path,
                     caption="ROC Curves — All Models",
                     use_container_width=True)

    with img_col4:
        cm_path = os.path.join(base_path, 'confusion_matrices.png')
        if os.path.exists(cm_path):
            st.image(cm_path,
                     caption="Confusion Matrices — All Models",
                     use_container_width=True)


# ════════════════════════════════════════════════════════════
#  TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📖 About This Project")
    st.markdown("""
    ### Employee Attrition Predictor

    This application predicts the likelihood of an employee leaving a company
    using machine learning, built on the **IBM HR Analytics dataset**.

    ---

    ### 🔧 Technical Stack
    - **Data Processing**: pandas, numpy, scikit-learn
    - **Imbalance Handling**: SMOTE (imbalanced-learn)
    - **Models**: Logistic Regression, Random Forest, XGBoost
    - **Interpretability**: SHAP (SHapley Additive exPlanations)
    - **App**: Streamlit

    ---

    ### 🧠 Methodology
    1. **EDA** — Explored 35 features across 1,470 employees
    2. **Preprocessing** — Encoding, scaling, SMOTE for class imbalance
    3. **Feature Engineering** — 7 new domain-driven features created
    4. **Modeling** — 3 models trained with cross-validation
    5. **Interpretability** — SHAP explanations at global and individual level
    6. **Deployment** — Interactive Streamlit app

    ---

    ### 📊 Dataset
    IBM HR Analytics Employee Attrition Dataset
    — 1,470 employees, 35 features, 16% attrition rate

    ---

    ### 👤 Built by
    **Emad Kalantari** — [GitHub Repository](https://github.com/emaadkalantarii/employee-attrition)
    """)
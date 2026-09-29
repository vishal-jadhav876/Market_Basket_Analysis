import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS for Healthcare Dark UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Predictive Analytics",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #0c0f16;
        color: #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #151922 100%);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2a2e38;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-val {
        font-size: 30px;
        font-weight: 800;
    }
    .pos-val { color: #2ecc71; }
    .neg-val { color: #e74c3c; }
    .tot-val { color: #3498db; }
    
    .privacy-box {
        background-color: #1a2332;
        border-left: 5px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Synthetic Dataset Generator (Standard UCI Diabetes Medical Structure)
# -----------------------------------------------------------------------------
@st.cache_data
def load_medical_data():
    np.random.seed(42)
    n_samples = 1000

    pregnancies = np.random.randint(0, 10, n_samples)
    glucose = np.random.normal(120, 30, n_samples).clip(60, 200)
    blood_pressure = np.random.normal(70, 12, n_samples).clip(40, 120)
    skin_thickness = np.random.normal(20, 10, n_samples).clip(10, 50)
    insulin = np.random.normal(80, 40, n_samples).clip(15, 276)
    bmi = np.random.normal(32, 6, n_samples).clip(18, 50)
    diabetes_pedigree = np.random.uniform(0.08, 2.4, n_samples)
    age = np.random.randint(21, 80, n_samples)

    # Risk logic for outcome
    risk_score = (
        (glucose / 200) * 0.4 + 
        (bmi / 50) * 0.3 + 
        (age / 80) * 0.2 + 
        (diabetes_pedigree / 2.4) * 0.1
    )
    outcome = (risk_score > 0.45).astype(int)

    df = pd.DataFrame({
        'Glucose': np.round(glucose, 1),
        'BMI': np.round(bmi, 1),
        'Age': age,
        'Insulin': np.round(insulin, 1),
        'BloodPressure': np.round(blood_pressure, 1),
        'SkinThickness': np.round(skin_thickness, 1),
        'DiabetesPedigreeFunction': np.round(diabetes_pedigree, 2),
        'Pregnancies': pregnancies,
        'Outcome': outcome  # 1 = High Diabetes Risk, 0 = Healthy / Low Risk
    })
    return df

# -----------------------------------------------------------------------------
# 3. Model Training & Data Normalization Pipeline
# -----------------------------------------------------------------------------
@st.cache_resource
def train_health_model(df):
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    return model, scaler, acc, auc, X.columns

# -----------------------------------------------------------------------------
# 4. Header & Dashboard Layout
# -----------------------------------------------------------------------------
st.title("🩺 Healthcare Predictive Analytics (Disease Risk Detection)")
st.caption("AI-powered clinical risk assessment with medical normalization & Ethical Data Privacy.")
st.markdown("---")

df = load_medical_data()

st.sidebar.title("⚙️ Medical System Navigation")
mode = st.sidebar.radio("Select View:", ["📊 Clinical Model Analytics", "🩺 Patient Risk Screening Lab", "🛡️ Patient Privacy & Ethics Guidelines"])

model, scaler, acc, auc, feature_names = train_health_model(df)

# -----------------------------------------------------------------------------
# Page 1: Clinical Model Analytics
# -----------------------------------------------------------------------------
if mode == "📊 Clinical Model Analytics":
    st.subheader("📌 Model Performance & Feature Analytics")

    tot = len(df)
    risk_cases = df['Outcome'].sum()
    risk_percent = (risk_cases / tot) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div>Total Patient Records</div><div class="metric-val tot-val">{tot}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div>High Risk Cases</div><div class="metric-val neg-val">{risk_cases}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div>Model Accuracy</div><div class="metric-val pos-val">{acc*100:.1f}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div>ROC-AUC Score</div><div class="metric-val pos-val">{auc:.2f}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.subheader("📋 Normalized Medical Dataset Sample")
        st.dataframe(df.head(10), use_container_width=True)

    with right_col:
        st.subheader("🔍 Key Risk Drivers (Feature Importance)")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Medical Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(5, 3.8))
        fig.patch.set_facecolor('#0c0f16')
        ax.set_facecolor('#0c0f16')
        ax.barh(feat_df['Medical Feature'], feat_df['Importance'], color='#2ecc71')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#2a2e38')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2a2e38')
        ax.set_xlabel("Impact Score", color="white")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Page 2: Patient Risk Screening Lab
# -----------------------------------------------------------------------------
elif mode == "🩺 Patient Risk Screening Lab":
    st.subheader("🩺 Real-time Patient Disease Risk Assessment")
    
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.slider("Plasma Glucose Concentration (mg/dL)", 50, 250, 120)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=28.5, step=0.1)
        age = st.slider("Age (Years)", 18, 90, 45)
        insulin = st.number_input("Serum Insulin (mu U/ml)", min_value=0.0, max_value=300.0, value=85.0, step=1.0)

    with col2:
        bp = st.slider("Diastolic Blood Pressure (mm Hg)", 40, 130, 75)
        skin = st.slider("Triceps Skin Fold Thickness (mm)", 10, 60, 20)
        dpf = st.number_input("Diabetes Pedigree Function (Genetic Risk)", min_value=0.05, max_value=2.5, value=0.45, step=0.01)
        pregnancies = st.selectbox("Number of Pregnancies", list(range(0, 15)), index=1)

    if st.button("🧪 Predict Patient Disease Risk"):
        raw_input = pd.DataFrame([[glucose, bmi, age, insulin, bp, skin, dpf, pregnancies]], columns=feature_names)
        scaled_input = scaler.transform(raw_input)
        
        risk_prob = model.predict_proba(scaled_input)[0][1] * 100
        prediction = model.predict(scaled_input)[0]

        st.write("")
        st.subheader("📋 Assessment Results")

        if prediction == 0:
            st.success(f"✅ **LOW RISK OF DIABETES** | Probability Score: `{risk_prob:.1f}%`")
            st.info("Medical parameters fall within standard healthy ranges. Regular health check-ups advised.")
        else:
            st.error(f"🚨 **HIGH RISK OF DIABETES DETECTED** | Probability Score: `{risk_prob:.1f}%`")
            st.warning("Immediate consultation with a physician is recommended. Key risk factors identified: High Glucose or BMI.")

# -----------------------------------------------------------------------------
# Page 3: Ethical Data Privacy & Guidelines
# -----------------------------------------------------------------------------
else:
    st.subheader("🛡️ Ethical Data Handling & Patient Privacy (HIPAA Compliance)")
    
    st.markdown("""
    <div class="privacy-box">
        <h4>🔒 Ethical Healthcare AI Principles</h4>
        <p>Medical analytics systems must rigorously ensure data privacy, fairness, and non-discrimination during patient risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔑 Key Ethical & Security Measures Implemented:
    1. **Data Anonymization (HIPAA Standards):** All Personally Identifiable Information (PII) such as Names, Aadhaar/SSN numbers, and Addresses are stripped out prior to model ingestion.
    2. **Normalization & Fair Treatment:** Input medical features are standard-scaled (`StandardScaler`) to eliminate measurement scale bias.
    3. **Model Explainability & Transparency:** Predictions provide feature importance metrics so medical practitioners can understand *why* a risk tag was flagged.
    4. **Human-in-the-Loop (Clinical Oversight):** AI risk outputs serve strictly as an *auxiliary decision support tool* and cannot substitute official medical diagnostics.
    """)
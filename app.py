import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Job Posting Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load models
# -------------------------------------------------
@st.cache_resource
def load_models():
    try:
        models = {}
        model_names = [
            'logistic_regression',
            'decision_tree',
            'k-nearest_neighbors',
            'naive_bayes',
            'random_forest',
            'xgboost'
        ]

        model_dir = 'model'

        for name in model_names:
            filepath = os.path.join(model_dir, f'{name}.pkl')
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    models[name] = pickle.load(f)

        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)

        with open(os.path.join(model_dir, 'label_encoders.pkl'), 'rb') as f:
            label_encoders = pickle.load(f)

        with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)

        return models, scaler, label_encoders, feature_names, True

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Train models first using train_models.ipynb")
        return {}, None, {}, [], False


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


def preprocess_input_data(df, label_encoders):
    try:
        df_processed = df.copy()
        categorical_cols = [
            'employment_type',
            'required_experience',
            'required_education',
            'industry',
            'function'
        ]

        for col in categorical_cols:
            if col in df_processed.columns and col in label_encoders:
                le = label_encoders[col]
                df_processed[col] = df_processed[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

        return df_processed, True

    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None, False


# -------------------------------------------------
# Load Everything
# -------------------------------------------------
models, scaler, label_encoders, feature_names, models_loaded = load_models()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<h1 class="main-header">Job Posting Fraud Detection System</h1>',
            unsafe_allow_html=True)

if models_loaded:
    st.success(f"✓ {len(models)} Models Loaded Successfully")
else:
    st.error("✗ Models not loaded. Run train_models.ipynb first.")

st.markdown("---")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Single Prediction",
    "Batch Prediction (CSV)",
    "About"
])

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
if page == "Home":

    st.markdown('<h2 class="sub-header">Welcome</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("17,880 Job Postings\n12 Features Used")

    with col2:
        st.success(f"{len(models)} ML Models\nTrained & Ready")

    with col3:
        st.warning("CSV Upload\nBatch Predictions")

    st.markdown("---")

    st.subheader("Model Performance Overview")

    image_path = os.path.join("model", "confusion_matrices.png")

    if os.path.exists(image_path):
        st.image(
        image_path,
        caption="Confusion Matrix - Best Model",
        width='stretch'   # replaces use_column_width
    )
    else:
        st.warning("Confusion matrix image not found in model folder.")

# -------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------
elif page == "Single Prediction":

    if not models_loaded:
        st.stop()

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            telecommuting = st.selectbox("Telecommuting", [0, 1])
            has_logo = st.selectbox("Has Company Logo", [0, 1])
            has_questions = st.selectbox("Has Screening Questions", [0, 1])
            emp_type = st.selectbox("Employment Type",
                                    ['Full-time', 'Part-time', 'Contract', 'Temporary'])
            exp = st.selectbox("Required Experience",
                               ['Entry level', 'Mid-Senior level', 'Executive', 'Internship'])
            edu = st.selectbox("Required Education",
                               ["Bachelor's Degree", "Master's Degree",
                                "High School or equivalent", "Unspecified"])

        with col2:
            industry = st.text_input("Industry", "Information Technology")
            function = st.text_input("Job Function", "Engineering")
            has_profile = st.selectbox("Has Company Profile", [0, 1])
            has_desc = st.selectbox("Has Description", [0, 1])
            has_req = st.selectbox("Has Requirements", [0, 1])
            has_ben = st.selectbox("Has Benefits", [0, 1])

        model_choice = st.selectbox("Select Model", list(models.keys()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame({
            'telecommuting': [telecommuting],
            'has_company_logo': [has_logo],
            'has_questions': [has_questions],
            'employment_type': [emp_type],
            'required_experience': [exp],
            'required_education': [edu],
            'industry': [industry],
            'function': [function],
            'has_company_profile': [has_profile],
            'has_description': [has_desc],
            'has_requirements': [has_req],
            'has_benefits': [has_ben]
        })

        input_proc, success = preprocess_input_data(input_df, label_encoders)

        if success:
            model = models[model_choice]
            pred = model.predict(input_proc)[0]
            proba = model.predict_proba(input_proc)[0]

            if pred == 1:
                st.error("FRAUDULENT JOB POSTING")
            else:
                st.success("LEGITIMATE JOB POSTING")

            st.metric("Confidence", f"{proba[pred]*100:.2f}%")

# -------------------------------------------------
# BATCH PREDICTION
# -------------------------------------------------
elif page == "Batch Prediction (CSV)":

    if not models_loaded:
        st.stop()

    # ✅ Download real test.csv
    test_file_path = "test.csv"

    if os.path.exists(test_file_path):
        with open(test_file_path, "rb") as file:
            st.download_button(
                label="Download Test CSV",
                data=file,
                file_name="test.csv",
                mime="text/csv"
            )
    else:
        st.warning("test.csv not found in project directory.")

    uploaded = st.file_uploader("Upload CSV File", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        model_choice = st.selectbox("Select Model", list(models.keys()))

        if st.button("Run Prediction"):

            df_proc, success = preprocess_input_data(df, label_encoders)

            if success:
                model = models[model_choice]
                preds = model.predict(df_proc)
                probas = model.predict_proba(df_proc)

                result_df = df.copy()
                result_df['Prediction'] = preds
                result_df['Prediction_Label'] = result_df['Prediction'].map({0: 'Real', 1: 'Fake'})
                result_df['Confidence_Real'] = probas[:, 0]
                result_df['Confidence_Fake'] = probas[:, 1]

                st.success("Prediction Completed")
                st.dataframe(result_df)

                # Confusion Matrix
                if 'fraudulent' in df.columns:
                    actual = df['fraudulent'].values
                    cm = confusion_matrix(actual, preds)

                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicted Real', 'Predicted Fake'],
                        y=['Actual Real', 'Actual Fake'],
                        text=cm,
                        texttemplate='%{text}',
                        colorscale='Blues'
                    ))
                    st.plotly_chart(fig)

                st.download_button(
                    "Download Predictions",
                    convert_df_to_csv(result_df),
                    f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# -------------------------------------------------
# ABOUT
# -------------------------------------------------
elif page == "About":
    st.markdown("""
    ML-based Job Posting Fraud Detection System.

    Models Used:
    - Logistic Regression
    - Decision Tree
    - KNN
    - Naive Bayes
    - Random Forest
    - XGBoost

    Built using Streamlit + Scikit-learn.
    """)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown("Job Posting Fraud Detection System | ML Assignment 2026")

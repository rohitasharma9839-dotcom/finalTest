import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder

# ===========================
# 1. Page Config & Setup
# ===========================
st.set_page_config(page_title="Model Evaluator", layout="wide")
st.title("🛒 Model Performance Evaluator")

# Github URL for test data
csv_url = "https://raw.githubusercontent.com/rohitasharma9839-dotcom/finalTest/refs/heads/main/adult_test.csv"

# ===========================
# 2. Shared Preprocessing Logic
# ===========================
def preprocess_test_data(df):
    # Strip whitespace and handle '?'
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Standardize target (assumes last column)
    target_col = df.columns[-1]
    df.rename(columns={target_col: 'target'}, inplace=True)
    
    # Clean target string (remove dots)
    df['target'] = df['target'].astype(str).str.replace('.', '', regex=False)
    
    # Label Encoding for categorical columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
            
    return df

# ===========================
# 3. Sidebar & Selection
# ===========================
model_map = {
    "Logistic Regression": "lr_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "KNN": "knn_model.pkl",
    "Naive Bayes": "nb_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

selected_model = st.sidebar.selectbox("Select Model", list(model_map.keys()))

# ===========================
# 4. Main Execution
# ===========================
try:
    # Load and Preprocess Test Data
    raw_test_df = pd.read_csv(csv_url)
    st.write("### Test Dataset Preview")
    st.dataframe(raw_test_df.head())
    
    if st.button(f"Evaluate {selected_model}"):
        # Process data
        df_clean = preprocess_test_data(raw_test_df.copy())
        X_test = df_clean.drop('target', axis=1)
        y_true = df_clean['target'].astype(int)

        # Load Pickle
        model_path = f"models/{model_map[selected_model]}"
        model = joblib.load(model_path)

        # Scaling for distance-based models
        if selected_model in ["Logistic Regression", "KNN"]:
            scaler = joblib.load('models/scaler.pkl')
            X_test = scaler.transform(X_test)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Metrics display
        st.subheader(f"📊 {selected_model} Metrics")
        cols = st.columns(6)
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted'),
            "Recall": recall_score(y_true, y_pred, average='weighted'),
            "F1": f1_score(y_true, y_pred, average='weighted')
        }
        
        for i, (label, val) in enumerate(metrics.items()):
            cols[i].metric(label, f"{val:.4f}")

        # Visualizations
        st.write("---")
        vcol1, vcol2 = st.columns(2)
        
        with vcol1:
            st.write("#### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
            

        with vcol2:
            st.write("#### Classification Report")
            report = classification_report(y_true, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

except Exception as e:

    st.error(f"Waiting for test data or model files... Error: {e}")

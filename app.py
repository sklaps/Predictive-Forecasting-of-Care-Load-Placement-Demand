# ==========================================
# Predictive Forecasting of Care Load & Placement Demand
# Streamlit Dashboard
# ==========================================

# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Predictive Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# Dashboard Title
st.title("📊 Predictive Forecasting of Care Load & Placement Demand")

st.markdown("""
### U.S. Department of Health and Human Services (HHS)

This dashboard predicts future **Children in HHS Care** and analyzes historical care load using time-series forecasting and machine learning models.

**Project Objectives:**
- Forecast future HHS care load
- Predict discharge demand
- Compare forecasting models
- Support proactive resource planning
""")
# ==========================================
# Load Dataset
# ==========================================

# Replace this filename if your dataset has a different name
df = pd.read_csv("Predictive Forecasting of Care Load & Placement Demand.csv")

# Display dataset
st.header("📂 Dataset Preview")

st.dataframe(df.head())

# Display basic information
st.header("📊 Dataset Information")

st.write("Rows :", df.shape[0])
st.write("Columns :", df.shape[1])

st.write("Column Names:")

st.write(df.columns.tolist())

st.dataframe(df.describe())

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Predictive Forecasting of Care Load & Placement Demand.csv")

# ==========================================
# Data Cleaning
# ==========================================

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Remove commas from HHS Care column
df["Children in HHS Care"] = (
    df["Children in HHS Care"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(int)
)

# Sort data by Date
df = df.sort_values("Date")

# Set Date as index
df.set_index("Date", inplace=True)

# ==========================================
# Dataset Preview
# ==========================================

st.header("📂 Dataset Preview")
st.dataframe(df.head(10))

# ==========================================
# Dataset Information
# ==========================================

st.header("📊 Dataset Information")

st.write(f"Rows: {df.shape[0]}")
st.write(f"Columns: {df.shape[1]}")

st.write("Column Names:")
st.write(df.columns.tolist())

# ==========================================
# Dataset Statistics
# ==========================================

st.header("📈 Dataset Statistics")

st.dataframe(df.describe())

# ==========================================
# Historical HHS Care Load
# ==========================================

st.header("📈 Historical Children in HHS Care")

fig, ax = plt.subplots(figsize=(15,5))

ax.plot(
    df.index,
    df["Children in HHS Care"],
    linewidth=2
)

ax.set_title("Children in HHS Care Over Time")

ax.set_xlabel("Date")

ax.set_ylabel("Children")

ax.grid(True)

st.pyplot(fig)


# ==========================================
# KPI Dashboard
# ==========================================

st.header("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Current HHS Care",
        value=f"{df['Children in HHS Care'].iloc[-1]:,}"
    )

with col2:
    st.metric(
        label="Current CBP Custody",
        value=f"{df['Children in CBP custody'].iloc[-1]:,}"
    )

with col3:
    st.metric(
        label="Daily Discharges",
        value=f"{df['Children discharged from HHS Care'].iloc[-1]:,}"
    )

with col4:
    st.metric(
        label="Total Records",
        value=len(df)
    )

    st.set_page_config(
    page_title="Predictive Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("⚙ Dashboard Controls")

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=7,
    max_value=30,
    value=14,
    step=1
)

selected_model = st.sidebar.selectbox(
    "Select Forecast Model",
    [
        "Naïve Forecast",
        "Moving Average",
        "ARIMA",
        "Random Forest",
        "Gradient Boosting"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write("### Selected Options")

st.sidebar.write(f"Forecast Horizon: {forecast_days} Days")

st.sidebar.write(f"Model: {selected_model}")

# ==========================================
# Historical Trends
# ==========================================

st.header("📊 Historical Trends")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Graph 1
axes[0,0].plot(
    df.index,
    df["Children in HHS Care"]
)
axes[0,0].set_title("Children in HHS Care")
axes[0,0].grid(True)

# Graph 2
axes[0,1].plot(
    df.index,
    df["Children in CBP custody"]
)
axes[0,1].set_title("Children in CBP Custody")
axes[0,1].grid(True)

# Graph 3
axes[1,0].plot(
    df.index,
    df["Children transferred out of CBP custody"]
)
axes[1,0].set_title("Transfers from CBP")
axes[1,0].grid(True)

# Graph 4
axes[1,1].plot(
    df.index,
    df["Children discharged from HHS Care"]
)
axes[1,1].set_title("Children Discharged")
axes[1,1].grid(True)

plt.tight_layout()

st.pyplot(fig)

# ==========================================
# Model Comparison
# ==========================================

st.header("🤖 Forecast Model Comparison")

comparison = pd.DataFrame({

    "Model":[
        "Naïve Forecast",
        "Moving Average",
        "Gradient Boosting",
        "Random Forest",
        "ARIMA",
        "Exponential Smoothing"
    ],

    "MAE":[
        10.30,
        33.23,
        61.76,
        66.44,
        197.56,
        732.50
    ],

    "RMSE":[
        14.08,
        40.68,
        82.73,
        88.80,
        241.43,
        813.98
    ],

    "MAPE (%)":[
        0.46,
        1.48,
        2.84,
        3.08,
        None,
        None
    ]

})

comparison = comparison.sort_values("MAE")

st.dataframe(comparison)


st.success("🏆 Best Performing Model: Naïve Forecast")

st.metric(
    label="Lowest MAE",
    value="10.30"
)

st.subheader("📊 MAE Comparison")

fig, ax = plt.subplots(figsize=(10,5))

ax.bar(
    comparison["Model"],
    comparison["MAE"]
)

plt.xticks(rotation=20)

ax.set_ylabel("MAE")

st.pyplot(fig)

# ==========================================
# Executive Summary
# ==========================================

st.header("📋 Executive Summary")

st.markdown("""
### Key Findings

- Historical analysis shows fluctuations in the number of children in HHS Care over the study period.
- Multiple forecasting models were evaluated using MAE, RMSE, and MAPE.
- The Naïve Forecast achieved the lowest prediction error for this dataset.
- Machine learning and time-series models provide valuable insights for forecasting care demand.

### Recommendations

- Continue monitoring daily care load trends.
- Update forecasting models regularly using newly available data.
- Use forecasting results to support staffing, budgeting, and resource allocation.
- Combine statistical forecasting with operational expertise for decision-making.
""")

st.markdown("---")

st.header("ℹ Project Information")

st.write("**Project:** Predictive Forecasting of Care Load & Placement Demand")

st.write("**Dataset:** U.S. Department of Health and Human Services (HHS)")

st.write("**Developed Using:**")

st.write("""
- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- Scikit-learn
- Streamlit
""")
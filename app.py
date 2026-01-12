import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# ==============================================================================
# 0. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Disbursement Forecast AI", layout="wide")

# Custom CSS for styling metrics
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #5DADE2; }
    .main { background-color: #0E1117; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Daily Disbursement Intelligence System")
st.markdown("Automated AI Forecasting for Q1 2026 using Random Forest Regression")

# ==============================================================================
# 1. DATA LOADING & SIDEBAR
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    # Path for local machine
    LOCAL_PATH = r'D:\Data Science\Output Data\DISB_Updated_query.xlsx'
    
    if os.path.exists(LOCAL_PATH):
        st.info(f"Connected to local source: {os.path.basename(LOCAL_PATH)}")
        uploaded_file = LOCAL_PATH
    else:
        uploaded_file = st.file_uploader("Upload 'DISB_Updated_query.xlsx'", type=['xlsx'])
    
    max_disb_cap = st.number_input("Business Cap (Units)", value=9000000000)
    forecast_days = st.slider("Forecast Horizon (Days)", 30, 90, 90)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df = df[['DISBURSMENT_DATE', 'BRANCH_CODE', 'DISBURSED_AMOUNT']].copy()
        df['DISBURSMENT_DATE'] = pd.to_datetime(df['DISBURSMENT_DATE'])
        st.toast("Data Loaded Successfully!", icon="✅")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # ==============================================================================
    # 2. PROCESSING & MODEL TRAINING
    # ==============================================================================
    with st.spinner('AI Engine processing trends...'):
        # Aggregation & Anomaly Detection
        df_agg = df.groupby(['DISBURSMENT_DATE', 'BRANCH_CODE'])['DISBURSED_AMOUNT'].sum().reset_index()
        df_agg = df_agg.rename(columns={'DISBURSED_AMOUNT': 'TOTAL_DAILY_DISBURSEMENT'})
        
        df_agg['DAILY_MEAN'] = df_agg.groupby('BRANCH_CODE')['TOTAL_DAILY_DISBURSEMENT'].transform('mean')
        df_agg['DAILY_STD'] = df_agg.groupby('BRANCH_CODE')['TOTAL_DAILY_DISBURSEMENT'].transform('std')
        df_agg['IS_ANOMALY'] = (df_agg['TOTAL_DAILY_DISBURSEMENT'] > df_agg['DAILY_MEAN'] + 3 * df_agg['DAILY_STD'])

        # Feature Engineering
        def create_features(data):
            data = data.sort_values(by='DISBURSMENT_DATE')
            data['day_of_week'] = data['DISBURSMENT_DATE'].dt.dayofweek
            data['month'] = data['DISBURSMENT_DATE'].dt.month
            data['day_of_year'] = data['DISBURSMENT_DATE'].dt.dayofyear
            data['lag_1'] = data['TOTAL_DAILY_DISBURSEMENT'].shift(1)
            data['lag_7'] = data['TOTAL_DAILY_DISBURSEMENT'].shift(7)
            data['rolling_mean_7'] = data['TOTAL_DAILY_DISBURSEMENT'].shift(1).rolling(window=7).mean()
            return data.dropna()

        df_model = df_agg.groupby('BRANCH_CODE', group_keys=False).apply(create_features).reset_index(drop=True)
        le = LabelEncoder()
        df_model['branch_code_encoded'] = le.fit_transform(df_model['BRANCH_CODE'])

        # Training
        FEATURES = ['branch_code_encoded', 'day_of_week', 'month', 'day_of_year', 'lag_1', 'lag_7', 'rolling_mean_7']
        X = df_model[FEATURES]
        y_log = np.log1p(df_model['TOTAL_DAILY_DISBURSEMENT'])
        
        reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        reg.fit(X, y_log)
        
        # Performance calculation (simplified)
        mae = mean_absolute_error(df_model['TOTAL_DAILY_DISBURSEMENT'], np.expm1(reg.predict(X)))

    # ==============================================================================
    # 3. RECURSIVE FORECASTING
    # ==============================================================================
    last_date = df_model['DISBURSMENT_DATE'].max()
    start_date = last_date + timedelta(days=1)
    all_branches = df_model['BRANCH_CODE'].unique()
    le_map = dict(zip(le.classes_, le.transform(le.classes_)))

    forecast_results = []
    for branch in all_branches:
        last_obs = df_model[df_model['BRANCH_CODE'] == branch].iloc[-1]
        c_l1, c_l7, c_rm7 = last_obs['TOTAL_DAILY_DISBURSEMENT'], last_obs['lag_7'], last_obs['rolling_mean_7']
        
        for i in range(forecast_days):
            date = start_date + timedelta(days=i)
            input_data = pd.DataFrame([{
                'branch_code_encoded': le_map[branch], 'day_of_week': date.dayofweek,
                'month': date.month, 'day_of_year': date.dayofyear,
                'lag_1': c_l1, 'lag_7': c_l7, 'rolling_mean_7': c_rm7
            }])[FEATURES]
            
            pred = np.expm1(reg.predict(input_data)[0])
            c_l1 = pred
            forecast_results.append({'DISBURSMENT_DATE': date, 'BRANCH_CODE': branch, 'Predicted_Disbursement': pred})

    df_forecast = pd.DataFrame(forecast_results)
    df_forecast['Predicted_Disbursement'] = df_forecast['Predicted_Disbursement'].clip(0, max_disb_cap)

    # ==============================================================================
    # 4. DASHBOARD - EXECUTIVE KPIS
    # ==============================================================================
    total_q1 = df_forecast['Predicted_Disbursement'].sum()
    avg_daily = df_forecast['Predicted_Disbursement'].mean()
    peak_val = df_forecast['Predicted_Disbursement'].max()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Projected Q1 Total", f"${total_q1/1e9:,.2f} Bn")
    k2.metric("Avg Daily Need", f"${avg_daily/1e6:,.1f} M")
    k3.metric("Peak Day Forecast", f"${peak_val/1e6:,.1f} M")
    k4.metric("MAE (Model Precision)", f"${mae/1e3:,.1f} K")
    st.divider()

    # ==============================================================================
    # 5. DASHBOARD - MULTI-GRAPH ANALYTICS
    # ==============================================================================
    g1, g2 = st.columns(2)

    with g1:
        # GRAPH 1: Main Trend
        st.subheader("Total Daily Volume Trend")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        plt.style.use('dark_background')
        hist_total = df_agg.groupby('DISBURSMENT_DATE')['TOTAL_DAILY_DISBURSEMENT'].sum()
        fore_total = df_forecast.groupby('DISBURSMENT_DATE')['Predicted_Disbursement'].sum()
        ax1.plot(hist_total.index, hist_total.values, color='#5DADE2', label='2025 Actual', alpha=0.5)
        ax1.plot(fore_total.index, fore_total.values, color='#E74C3C', linestyle='--', label='2026 AI Forecast')
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e6:,.0f}M'))
        ax1.legend()
        st.pyplot(fig1)

        # GRAPH 2: Weekday Analysis
        st.subheader("Volume by Weekday")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        df_forecast['DayName'] = df_forecast['DISBURSMENT_DATE'].dt.day_name()
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow = df_forecast.groupby('DayName')['Predicted_Disbursement'].mean().reindex(order)
        dow.plot(kind='bar', ax=ax2, color='#2ECC71')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e6:,.1f}M'))
        st.pyplot(fig2)

    with g2:
        # GRAPH 3: Top Branches
        st.subheader("Top 10 Branches (Projected)")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        top10 = df_forecast.groupby('BRANCH_CODE')['Predicted_Disbursement'].sum().nlargest(10).sort_values()
        top10.plot(kind='barh', ax=ax3, color='#F4D03F')
        ax3.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e6:,.0f}M'))
        st.pyplot(fig3)

        # GRAPH 4: Cumulative Cash
        st.subheader("Cumulative Liquidity Burn")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        cum_sum = fore_total.cumsum()
        ax4.fill_between(cum_sum.index, cum_sum.values, color='#9B59B6', alpha=0.3)
        ax4.plot(cum_sum.index, cum_sum.values, color='#9B59B6', linewidth=2)
        ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e9:,.2f}Bn'))
        st.pyplot(fig4)

    # ==============================================================================
    # 6. DATA EXPORT
    # ==============================================================================
    st.divider()
    with st.expander("📝 View Detailed Forecast Data Table"):
        st.dataframe(df_forecast.style.format({"Predicted_Disbursement": "${:,.2f}"}))

    csv = df_forecast.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export AI Report (CSV)", data=csv, file_name="Q1_2026_Forecast.csv")

else:
    st.warning("⚠️ Please upload the source Excel file in the sidebar to generate the intelligence report.")
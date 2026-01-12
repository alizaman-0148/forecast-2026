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

# Set Web Page Configuration
st.set_page_config(page_title="Disbursement Forecast AI", layout="wide")

st.title("📊 Daily Disbursement Forecasting System")
st.markdown("This AI model predicts daily branch disbursements for the next 90 days using Random Forest Regression.")

# ==============================================================================
# 0. CONFIGURATION & DATA LOADING
# ==============================================================================
with st.sidebar:
    st.header("Settings & Data")
    uploaded_file = st.file_uploader("Upload 'DISB_Updated_query.xlsx'", type=['xlsx'])
    max_disb_cap = st.number_input("Max Disbursement Cap (Units)", value=9000000000)

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df = df[['DISBURSMENT_DATE', 'BRANCH_CODE', 'DISBURSED_AMOUNT']].copy()
        df['DISBURSMENT_DATE'] = pd.to_datetime(df['DISBURSMENT_DATE'])
        st.success("✅ Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # ==============================================================================
    # 1. DATA CLEANING & ANOMALY DETECTION
    # ==============================================================================
    df_agg = df.groupby(['DISBURSMENT_DATE', 'BRANCH_CODE'])['DISBURSED_AMOUNT'].sum().reset_index()
    df_agg = df_agg.rename(columns={'DISBURSED_AMOUNT': 'TOTAL_DAILY_DISBURSEMENT'})
    df_agg = df_agg.sort_values(by=['BRANCH_CODE', 'DISBURSMENT_DATE']).reset_index(drop=True)

    df_agg['DAILY_MEAN'] = df_agg.groupby('BRANCH_CODE')['TOTAL_DAILY_DISBURSEMENT'].transform('mean')
    df_agg['DAILY_STD'] = df_agg.groupby('BRANCH_CODE')['TOTAL_DAILY_DISBURSEMENT'].transform('std')
    df_agg['IS_ANOMALY'] = (
        (df_agg['TOTAL_DAILY_DISBURSEMENT'] > df_agg['DAILY_MEAN'] + 3 * df_agg['DAILY_STD']) |
        (df_agg['TOTAL_DAILY_DISBURSEMENT'] < df_agg['DAILY_MEAN'] - 3 * df_agg['DAILY_STD'])
    )

    # ==============================================================================
    # 2. FEATURE ENGINEERING
    # ==============================================================================
    def create_time_series_features(data):
        data = data.sort_values(by='DISBURSMENT_DATE')
        data['day_of_week'] = data['DISBURSMENT_DATE'].dt.dayofweek
        data['month'] = data['DISBURSMENT_DATE'].dt.month
        data['day_of_year'] = data['DISBURSMENT_DATE'].dt.dayofyear
        data['lag_1'] = data['TOTAL_DAILY_DISBURSEMENT'].shift(1)
        data['lag_7'] = data['TOTAL_DAILY_DISBURSEMENT'].shift(7)
        data['rolling_mean_7'] = data['TOTAL_DAILY_DISBURSEMENT'].shift(1).rolling(window=7).mean()
        return data.dropna()

    df_model = df_agg.groupby('BRANCH_CODE', group_keys=False).apply(create_time_series_features).reset_index(drop=True)
    le = LabelEncoder()
    df_model['branch_code_encoded'] = le.fit_transform(df_model['BRANCH_CODE'])

    # ==============================================================================
    # 3. MODEL TRAINING
    # ==============================================================================
    FEATURES = ['branch_code_encoded', 'day_of_week', 'month', 'day_of_year', 'lag_1', 'lag_7', 'rolling_mean_7']
    TARGET = 'TOTAL_DAILY_DISBURSEMENT'

    X = df_model[FEATURES]
    y = df_model[TARGET]
    split_point = int(len(df_model) * 0.85)
    
    X_train, y_train = X.iloc[:split_point], y.iloc[:split_point]
    X_test, y_test = X.iloc[split_point:], y.iloc[split_point:]

    # Model Execution
    with st.spinner('Training AI Model...'):
        reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        reg.fit(X_train, np.log1p(y_train))
        
        y_pred = np.expm1(reg.predict(X_test))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Final train on all data
        reg.fit(X, np.log1p(y))

    # Metrics Display
    m1, m2 = st.columns(2)
    m1.metric("Model Error (MAE)", f"${mae:,.2f}")
    feature_importance = pd.Series(reg.feature_importances_, index=FEATURES).sort_values(ascending=False)
    m2.metric("Top Predictor", feature_importance.index[0])

    # ==============================================================================
    # 4. RECURSIVE FORECASTING
    # ==============================================================================
    FORECAST_DAYS = 90
    last_known_date = df_model['DISBURSMENT_DATE'].max()
    start_date = last_known_date + timedelta(days=1)
    all_branches = df_model['BRANCH_CODE'].unique()
    le_map = dict(zip(le.classes_, le.transform(le.classes_)))

    forecast_results = []
    for branch in all_branches:
        last_obs = df_model[df_model['BRANCH_CODE'] == branch].sort_values('DISBURSMENT_DATE', ascending=False).iloc[0]
        cur_l1, cur_l7, cur_rm7 = last_obs['TOTAL_DAILY_DISBURSEMENT'], last_obs['lag_7'], last_obs['rolling_mean_7']
        
        for i in range(FORECAST_DAYS):
            date = start_date + timedelta(days=i)
            input_data = pd.DataFrame([{
                'branch_code_encoded': le_map.get(branch), 'day_of_week': date.dayofweek,
                'month': date.month, 'day_of_year': date.dayofyear,
                'lag_1': cur_l1, 'lag_7': cur_l7, 'rolling_mean_7': cur_rm7
            }])[FEATURES]
            
            pred = np.expm1(reg.predict(input_data)[0])
            cur_l1 = pred # Update lag for next iteration
            forecast_results.append({'DISBURSMENT_DATE': date, 'BRANCH_CODE': branch, 'Predicted_Disbursement': pred})

    df_forecast_final = pd.DataFrame(forecast_results)
    df_forecast_final['Predicted_Disbursement'] = df_forecast_final['Predicted_Disbursement'].clip(lower=0, upper=max_disb_cap).round(2)

    # ==============================================================================
    # 5. VISUALIZATION
    # ==============================================================================
    st.subheader("Interactive Forecast Visualization")
    
    # Prep Plotting Data
    df_agg_grouped = df_agg.groupby('DISBURSMENT_DATE').agg(Actual=('TOTAL_DAILY_DISBURSEMENT', 'sum'), Anomaly=('IS_ANOMALY', 'any')).reset_index()
    df_forecast_grouped = df_forecast_final.groupby('DISBURSMENT_DATE')['Predicted_Disbursement'].sum().reset_index()
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df_agg_grouped['DISBURSMENT_DATE'], df_agg_grouped['Actual'], color='#5DADE2', label='Actual 2025')
    ax.plot(df_forecast_grouped['DISBURSMENT_DATE'], df_forecast_grouped['Predicted_Disbursement'], color='#E74C3C', linestyle=':', label='Forecast 2026')
    
    # Anomaly scatter
    anoms = df_agg_grouped[df_agg_grouped['Anomaly'] == True]
    ax.scatter(anoms['DISBURSMENT_DATE'], anoms['Actual'], color='#F4D03F', s=30, label='Anomalies')

    def currency_formatter(x, pos): return f'${x/1e6:1.1f}M' if x >= 1e6 else f'${x:,.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    ax.legend()
    st.pyplot(fig)

    # Download Button
    csv = df_forecast_final.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast Data", data=csv, file_name="forecast_2026.csv", mime="text/csv")

else:
    st.info("Please upload your Excel file in the sidebar to begin.")
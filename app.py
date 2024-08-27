import streamlit as st
import pandas as pd
import numpy as np

st.title('Growth Rate Analyzer')

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Data Preview:")
    st.write(df.head())
    
    # Allow user to select columns
    page_col = st.selectbox("Select the column for 'Pages':", df.columns)
    traffic_col = st.selectbox("Select the column for 'Traffic':", df.columns)
    
    # Allow user to select the time frame and date frame
    time_frame = st.selectbox("Select Time Frame:", ['1m', '6m', '1y', '2y', '5y', 'all'])
    date_frame = st.selectbox("Select Date Frame:", ['daily', 'weekly', 'monthly'])
    
    df[page_col] = pd.to_numeric(df[page_col], errors='coerce')
    df[traffic_col] = pd.to_numeric(df[traffic_col], errors='coerce')
    
    df['Page Growth Rate'] = df[page_col].pct_change() * 100
    df['Traffic Change Rate'] = df[traffic_col].pct_change() * 100

    st.write("Calculated Growth Rates:")
    st.write(df[['Week', 'Page Growth Rate', 'Traffic Change Rate']].dropna().head())

    st.header("Analysis")
    
    def analyze_growth(df, window_size):
        df[f"Page Growth {window_size}M MA"] = df['Page Growth Rate'].rolling(window=window_size).mean()
        df[f"Traffic Change {window_size}M MA"] = df['Traffic Change Rate'].rolling(window=window_size).mean()
        df_ma = df.dropna(subset=[f"Page Growth {window_size}M MA", f"Traffic Change {window_size}M MA"])

        correlation_ma = df_ma[[f"Page Growth {window_size}M MA", f"Traffic Change {window_size}M MA"]].corr().iloc[0, 1]
        
        stable_growth_mask = (df_ma[f"Page Growth {window_size}M MA"] >= -2) & (df_ma[f"Page Growth {window_size}M MA"] <= 6)
        rapid_growth_mask = df_ma[f"Page Growth {window_size}M MA"] > 10

        stable_growth_traffic_change = df_ma[stable_growth_mask][f"Traffic Change {window_size}M MA"].mean()
        rapid_growth_traffic_change = df_ma[rapid_growth_mask][f"Traffic Change {window_size}M MA"].mean()

        return correlation_ma, stable_growth_traffic_change, rapid_growth_traffic_change

    # Mapping time_frame to window sizes in months
    time_frame_mapping = {'1m': 1, '6m': 6, '1y': 12, '2y': 24, '5y': 60, 'all': len(df)}
    window_size = time_frame_mapping[time_frame]

    correlation, stable_growth_traffic, rapid_growth_traffic = analyze_growth(df.copy(), window_size)
    
    st.subheader(f"{window_size}-Period Moving Average:")
    st.write(f"Correlation: {correlation:.4f}")
    st.write(f"Stable Growth Traffic Change: {stable_growth_traffic:.2f}%")
    st.write(f"Rapid Growth Traffic Change: {rapid_growth_traffic:.2f}%")
    
    st.write("Based on these findings, the app has analyzed the moving averages and provided insights into how the twiddler algorithm might be reacting to different growth scenarios.")

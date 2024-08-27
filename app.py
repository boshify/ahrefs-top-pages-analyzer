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
    
    # Allow user to select the date frame
    date_frame = st.selectbox("Select Date Frame:", ['daily', 'weekly', 'monthly'])

    # Allow user to select the moving average window size
    max_window_size = len(df)
    window_size = st.slider(f"Select Moving Average Window ({date_frame})", min_value=1, max_value=max_window_size, value=3, step=1)
    
    df[page_col] = pd.to_numeric(df[page_col], errors='coerce')
    df[traffic_col] = pd.to_numeric(df[traffic_col], errors='coerce')
    
    df['Page Growth Rate'] = df[page_col].pct_change() * 100
    df['Traffic Change Rate'] = df[traffic_col].pct_change() * 100

    st.write("Calculated Growth Rates:")
    st.write(df[['Week', 'Page Growth Rate', 'Traffic Change Rate']].dropna().head())

    st.header("Analysis")
    
    def analyze_growth(df, window_size):
        if len(df) < window_size:
            st.warning(f"Not enough data to calculate a {window_size}-period moving average.")
            return None, None, None

        df[f"Page Growth {window_size}MA"] = df['Page Growth Rate'].rolling(window=window_size).mean()
        df[f"Traffic Change {window_size}MA"] = df['Traffic Change Rate'].rolling(window=window_size).mean()
        df_ma = df.dropna(subset=[f"Page Growth {window_size}MA", f"Traffic Change {window_size}MA"])

        if df_ma.empty:
            st.warning("Insufficient data after applying the moving average. Please try a shorter window size or adjust your data.")
            return None, None, None

        correlation_ma = df_ma[[f"Page Growth {window_size}MA", f"Traffic Change {window_size}MA"]].corr().iloc[0, 1]
        
        stable_growth_mask = (df_ma[f"Page Growth {window_size}MA"] >= -2) & (df_ma[f"Page Growth {window_size}MA"] <= 6)
        rapid_growth_mask = df_ma[f"Page Growth {window_size}MA"] > 10

        stable_growth_traffic_change = df_ma[stable_growth_mask][f"Traffic Change {window_size}MA"].mean()
        rapid_growth_traffic_change = df_ma[rapid_growth_mask][f"Traffic Change {window_size}MA"].mean()

        return correlation_ma, stable_growth_traffic_change, rapid_growth_traffic_change

    correlation, stable_growth_traffic, rapid_growth_traffic = analyze_growth(df.copy(), window_size)
    
    if correlation is not None:
        st.subheader(f"{window_size}-Period Moving Average ({date_frame}):")
        st.write(f"Correlation: {correlation:.4f}")
        st.write(f"Stable Growth Traffic Change: {stable_growth_traffic:.2f}%")
        st.write(f"Rapid Growth Traffic Change: {rapid_growth_traffic:.2f}%")
    else:
        st.error("Analysis could not be completed due to insufficient data or a calculation error.")
    
    st.write("Based on these findings, the app has analyzed the moving averages and provided insights into how the twiddler algorithm might be reacting to different growth scenarios.")


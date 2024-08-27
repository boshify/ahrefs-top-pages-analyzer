import streamlit as st
import pandas as pd
import numpy as np

st.title('Growth Rate Analyzer')

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Data Preview:")
    st.write(df.head())

    df['Weekly Page Growth Rate'] = df['Organic pages (Monthly volume - United States)'].pct_change() * 100
    df['Weekly Traffic Change Rate'] = df['Avg. organic traffic (Monthly volume - United States)'].pct_change() * 100

    st.write("Calculated Growth Rates:")
    st.write(df[['Week', 'Weekly Page Growth Rate', 'Weekly Traffic Change Rate']].dropna().head())

    st.header("Analysis")
    
    def analyze_growth(df, window_size):
        df[f"Page Growth {window_size}M MA"] = df['Weekly Page Growth Rate'].rolling(window=window_size).mean()
        df[f"Traffic Change {window_size}M MA"] = df['Weekly Traffic Change Rate'].rolling(window=window_size).mean()
        df_ma = df.dropna(subset=[f"Page Growth {window_size}M MA", f"Traffic Change {window_size}M MA"])

        correlation_ma = df_ma[[f"Page Growth {window_size}M MA", f"Traffic Change {window_size}M MA"]].corr().iloc[0, 1]
        
        stable_growth_mask = (df_ma[f"Page Growth {window_size}M MA"] >= -2) & (df_ma[f"Page Growth {window_size}M MA"] <= 6)
        rapid_growth_mask = df_ma[f"Page Growth {window_size}M MA"] > 10

        stable_growth_traffic_change = df_ma[stable_growth_mask][f"Traffic Change {window_size}M MA"].mean()
        rapid_growth_traffic_change = df_ma[rapid_growth_mask][f"Traffic Change {window_size}M MA"].mean()

        return correlation_ma, stable_growth_traffic_change, rapid_growth_traffic_change

    window_sizes = [3, 6, 12]
    for window_size in window_sizes:
        correlation, stable_growth_traffic, rapid_growth_traffic = analyze_growth(df.copy(), window_size)
        st.subheader(f"{window_size}-Month Moving Average:")
        st.write(f"Correlation: {correlation:.4f}")
        st.write(f"Stable Growth Traffic Change: {stable_growth_traffic:.2f}%")
        st.write(f"Rapid Growth Traffic Change: {rapid_growth_traffic:.2f}%")
        st.write("---")

    st.write("Based on these findings, the app has analyzed the moving averages and provided insights into how the twiddler algorithm might be reacting to different growth scenarios.")


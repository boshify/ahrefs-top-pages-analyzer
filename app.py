import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('Growth Rate Analyzer')

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Data Preview:")
    st.write(df.head())

    # Allow user to select columns
    date_col = st.selectbox("Select the column for 'Date':", df.columns)
    page_col = st.selectbox("Select the column for 'Pages':", df.columns)
    traffic_col = st.selectbox("Select the column for 'Traffic':", df.columns)
    
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Allow user to select the date frame
    date_frame = st.selectbox("Select Date Frame:", ['daily', 'weekly', 'monthly'])

    # Aggregate data based on the selected date frame
    if date_frame == 'weekly':
        df = df.resample('W-Mon', on=date_col).sum().reset_index().sort_values(by=date_col)
    elif date_frame == 'monthly':
        df = df.resample('M', on=date_col).sum().reset_index().sort_values(by=date_col)
    
    # Calculate growth rates including negative percentages
    df['Page Growth Rate'] = df[page_col].pct_change() * 100
    df['Traffic Change Rate'] = df[traffic_col].pct_change() * 100

    st.write("Calculated Growth Rates:")
    st.write(df[[date_col, 'Page Growth Rate', 'Traffic Change Rate']].dropna().head())

    st.header("Analysis")
    
    def analyze_growth(df, window_size):
        if len(df) < window_size:
            st.warning(f"Not enough data to calculate a {window_size}-period moving average.")
            return None, None, None, None, None

        df[f"Page Growth {window_size}MA"] = df['Page Growth Rate'].rolling(window=window_size).mean()
        df[f"Traffic Change {window_size}MA"] = df['Traffic Change Rate'].rolling(window=window_size).mean()
        df_ma = df.dropna(subset=[f"Page Growth {window_size}MA", f"Traffic Change {window_size}MA"])

        if df_ma.empty:
            st.warning("Insufficient data after applying the moving average. Please try a shorter window size or adjust your data.")
            return None, None, None, None, None

        correlation_ma = df_ma[[f"Page Growth {window_size}MA", f"Traffic Change {window_size}MA"]].corr().iloc[0, 1]
        
        # Calculating stable growth, rapid growth, and volatility during rapid growth
        stable_growth_mask = (df_ma[f"Page Growth {window_size}MA"] >= -2) & (df_ma[f"Page Growth {window_size}MA"] <= 6)
        rapid_growth_mask = df_ma[f"Page Growth {window_size}MA"] > 10

        stable_growth_traffic_change = df_ma[stable_growth_mask][f"Traffic Change {window_size}MA"].mean()
        rapid_growth_traffic_change = df_ma[rapid_growth_mask][f"Traffic Change {window_size}MA"].mean()
        rapid_growth_traffic_std = df_ma[rapid_growth_mask][f"Traffic Change {window_size}MA"].std()

        return correlation_ma, stable_growth_traffic_change, rapid_growth_traffic_change, rapid_growth_traffic_std, df_ma

    # Allow user to select the moving average window size
    max_window_size = len(df)
    window_size = st.slider(f"Select Moving Average Window ({date_frame})", min_value=1, max_value=max_window_size, value=3, step=1)

    correlation, stable_growth_traffic, rapid_growth_traffic, rapid_growth_std, df_ma = analyze_growth(df.copy(), window_size)
    
    if correlation is not None:
        st.subheader(f"{window_size}-Period Moving Average ({date_frame}):")
        st.write(f"Correlation: {correlation:.4f}")

        st.write("### Insights")
        if stable_growth_traffic is not None:
            st.write(f"**Stable Growth (between -2% and 6%)**: The average traffic change rate during periods of stable growth is {stable_growth_traffic:.2f}%. This indicates that the twiddler algorithm seems to reward stability, leading to consistent traffic changes during these periods.")

        if rapid_growth_traffic is not None and not np.isnan(rapid_growth_traffic):
            st.write(f"**Rapid Growth (above 10%)**: The average traffic change rate during periods of rapid growth is {rapid_growth_traffic:.2f}%, showing a penalty effect as traffic tends to decrease. This aligns with the idea that the algorithm penalizes sharp increases in page growth.")
        
        if rapid_growth_std is not None and not np.isnan(rapid_growth_std):
            st.write(f"**Volatility during Rapid Growth**: The standard deviation of the traffic change rate during rapid growth periods is {rapid_growth_std:.2f}%, suggesting high volatility and unpredictable traffic changes during these times.")

        # Summary of findings
        st.write("### Summary of Findings")
        if stable_growth_traffic > 0:
            st.write("Based on these findings, it appears the twiddler algorithm rewards growth stability in the range of -2% to 6% with positive traffic changes.")
        if rapid_growth_traffic < 0:
            st.write("The algorithm seems to penalize growth when it exceeds 10%, leading to a decrease in traffic. The volatility during these periods also suggests that the traffic response is unpredictable.")

        st.write("### Visualization")
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Page Growth Rate (%)', color='tab:blue')
        ax1.plot(df_ma[date_col], df_ma[f"Page Growth {window_size}MA"], color='tab:blue', label=f'Page Growth {window_size}MA')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Traffic Change Rate (%)', color='tab:red')
        ax2.plot(df_ma[date_col], df_ma[f"Traffic Change {window_size}MA"], color='tab:red', label=f'Traffic Change {window_size}MA')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.error("Analysis could not be completed due to insufficient data or a calculation error.")
    
    st.write("Based on these findings, the app has analyzed the moving averages and provided insights into how the twiddler algorithm might be reacting to different growth scenarios.")


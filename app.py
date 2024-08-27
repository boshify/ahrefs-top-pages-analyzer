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
        
        # Dynamically calculate stable and rapid growth thresholds
        stable_min = df_ma[f"Page Growth {window_size}MA"].quantile(0.25)
        stable_max = df_ma[f"Page Growth {window_size}MA"].quantile(0.75)
        rapid_growth_threshold = df_ma[f"Page Growth {window_size}MA"].quantile(0.90)

        # Calculating stable growth, rapid growth, and volatility during rapid growth
        stable_growth_mask = (df_ma[f"Page Growth {window_size}MA"] >= stable_min) & (df_ma[f"Page Growth {window_size}MA"] <= stable_max)
        rapid_growth_mask = df_ma[f"Page Growth {window_size}MA"] > rapid_growth_threshold

        stable_growth_traffic_change = df_ma[stable_growth_mask][f"Traffic Change {window_size}MA"].mean()
        rapid_growth_traffic_change = df_ma[rapid_growth_mask][f"Traffic Change {window_size}MA"].mean()
        rapid_growth_traffic_std = df_ma[rapid_growth_mask][f"Traffic Change {window_size}MA"].std()

        return correlation_ma, stable_growth_traffic_change, rapid_growth_traffic_change, rapid_growth_traffic_std, df_ma, stable_min, stable_max, rapid_growth_threshold

    # Allow user to select the moving average window size
    max_window_size = len(df)
    window_size = st.slider(f"Select Moving Average Window ({date_frame})", min_value=1, max_value=max_window_size, value=3, step=1)

    correlation, stable_growth_traffic, rapid_growth_traffic, rapid_growth_std, df_ma, stable_min, stable_max, rapid_growth_threshold = analyze_growth(df.copy(), window_size)
    
    if correlation is not None:
        st.subheader(f"{window_size}-Period Moving Average ({date_frame}):")
        st.write(f"Correlation: {correlation:.4f}")

        st.write("### Insights")
        if stable_growth_traffic is not None:
            st.write(f"**Stable Growth (between {stable_min:.2f}% and {stable_max:.2f}%)**: During periods where page growth remains within this stable range, the average traffic change rate is {stable_growth_traffic:.2f}%. This suggests that the twiddler algorithm rewards stable page growth, resulting in consistent increases in traffic.")

        if rapid_growth_traffic is not None and not np.isnan(rapid_growth_traffic):
            st.write(f"**Rapid Growth (above {rapid_growth_threshold:.2f}%)**: When page growth exceeds this threshold, the average traffic change rate is {rapid_growth_traffic:.2f}%. This indicates that rapid increases in page growth are associated with significant changes in traffic, potentially penalizing sharp increases in page growth.")

        if rapid_growth_std is not None and not np.isnan(rapid_growth_std):
            st.write(f"**Volatility during Rapid Growth**: The standard deviation of the traffic change rate during periods of rapid growth is {rapid_growth_std:.2f}%, indicating high volatility and suggesting that traffic responses are unpredictable during these times.")

        # Summary of findings
        st.write("### Summary of Findings")
        st.write(f"Based on these findings, it appears the twiddler algorithm rewards growth stability in the range of {stable_min:.2f}% to {stable_max:.2f}% with positive traffic changes. However, if page growth exceeds {rapid_growth_threshold:.2f}%, it is likely to reduce traffic by an average of {abs(rapid_growth_traffic):.2f}%, with a volatility of {rapid_growth_std:.2f}%.")

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

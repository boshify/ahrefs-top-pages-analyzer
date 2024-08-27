import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import math

st.title('Growth Rate Analyzer with Lagged Correlation, Page Sensitivity, and Traffic per Page')

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

    # Calculate Traffic per Page
    df['Traffic per Page'] = df[traffic_col] / df[page_col]

    # Normalize the growth rate by the total number of pages
    df['Normalized Page Growth Rate'] = df['Page Growth Rate'] / np.log1p(df[page_col])

    # Allow user to set a decay factor
    decay_factor = st.slider("Select Decay Factor", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    # Apply the decay factor to the traffic change rate
    df['Adjusted Traffic Change Rate'] = df['Traffic Change Rate'] * (1 - decay_factor * np.log1p(df[page_col]))

    st.write("Calculated Growth Rates:")
    st.write(df[[date_col, 'Page Growth Rate', 'Normalized Page Growth Rate', 'Traffic Change Rate', 'Traffic per Page', 'Adjusted Traffic Change Rate']].dropna().head())

    # Allow user to select the lag period
    lag_period = st.slider("Select Lag Period (in periods)", min_value=0, max_value=12, value=0, step=1)

    if lag_period > 0:
        df['Lagged Adjusted Traffic Change Rate'] = df['Adjusted Traffic Change Rate'].shift(lag_period)
        df['Lagged Traffic per Page'] = df['Traffic per Page'].shift(lag_period)
    else:
        df['Lagged Adjusted Traffic Change Rate'] = df['Adjusted Traffic Change Rate']
        df['Lagged Traffic per Page'] = df['Traffic per Page']

    st.header("Analysis")
    
    def analyze_growth(df, window_size):
        if len(df) < window_size:
            st.warning(f"Not enough data to calculate a {window_size}-period moving average.")
            return None, None, None, None, None

        df[f"Normalized Page Growth {window_size}MA"] = df['Normalized Page Growth Rate'].rolling(window=window_size).mean()
        df[f"Lagged Adjusted Traffic Change {window_size}MA"] = df['Lagged Adjusted Traffic Change Rate'].rolling(window=window_size).mean()
        df[f"Lagged Traffic per Page {window_size}MA"] = df['Lagged Traffic per Page'].rolling(window=window_size).mean()
        df_ma = df.dropna(subset=[f"Normalized Page Growth {window_size}MA", f"Lagged Adjusted Traffic Change {window_size}MA", f"Lagged Traffic per Page {window_size}MA"])

        if df_ma.empty:
            st.warning("Insufficient data after applying the moving average. Please try a shorter window size or adjust your data.")
            return None, None, None, None, None

        correlation_ma = df_ma[[f"Normalized Page Growth {window_size}MA", f"Lagged Adjusted Traffic Change {window_size}MA"]].corr().iloc[0, 1]
        
        # Dynamically calculate stable and rapid growth thresholds
        stable_min = df_ma[f"Normalized Page Growth {window_size}MA"].quantile(0.25)
        stable_max = df_ma[f"Normalized Page Growth {window_size}MA"].quantile(0.75)
        rapid_growth_threshold = df_ma[f"Normalized Page Growth {window_size}MA"].quantile(0.90)

        # Calculating stable growth, rapid growth, and volatility during rapid growth
        stable_growth_mask = (df_ma[f"Normalized Page Growth {window_size}MA"] >= stable_min) & (df_ma[f"Normalized Page Growth {window_size}MA"] <= stable_max)
        rapid_growth_mask = df_ma[f"Normalized Page Growth {window_size}MA"] > rapid_growth_threshold

        stable_growth_traffic_change = df_ma[stable_growth_mask][f"Lagged Adjusted Traffic Change {window_size}MA"].mean()
        rapid_growth_traffic_change = df_ma[rapid_growth_mask][f"Lagged Adjusted Traffic Change {window_size}MA"].mean()
        rapid_growth_traffic_std = df_ma[rapid_growth_mask][f"Lagged Adjusted Traffic Change {window_size}MA"].std()

        return correlation_ma, stable_growth_traffic_change, rapid_growth_traffic_change, rapid_growth_traffic_std, df_ma, stable_min, stable_max, rapid_growth_threshold

    # Allow user to select the moving average window size
    max_window_size = len(df)
    window_size = st.slider(f"Select Moving Average Window ({date_frame})", min_value=1, max_value=max_window_size, value=3, step=1)

    correlation, stable_growth_traffic, rapid_growth_traffic, rapid_growth_std, df_ma, stable_min, stable_max, rapid_growth_threshold = analyze_growth(df.copy(), window_size)
    
    if correlation is not None:
        st.subheader(f"{window_size}-Period Moving Average ({date_frame}):")
        st.write(f"Correlation (with {lag_period}-period lag): {correlation:.4f}")

        st.write("### Insights")
        if stable_growth_traffic is not None:
            st.write(f"**Stable Growth (between {stable_min:.2f}% and {stable_max:.2f}%)**: During periods where normalized page growth remains within this stable range, the average lagged and adjusted traffic change rate is {stable_growth_traffic:.2f}%. This suggests that the twiddler algorithm rewards stable page growth, resulting in consistent increases in traffic after a lag of {lag_period} periods.")

        if rapid_growth_traffic is not None and not np.isnan(rapid_growth_traffic):
            st.write(f"**Rapid Growth (above {rapid_growth_threshold:.2f}%)**: When normalized page growth exceeds this threshold, the average lagged and adjusted traffic change rate is {rapid_growth_traffic:.2f}%. This indicates that rapid increases in page growth are associated with significant changes in traffic, potentially penalizing sharp increases in page growth after a lag of {lag_period} periods.")

        if rapid_growth_std is not None and not np.isnan(rapid_growth_std):
            st.write(f"**Volatility during Rapid Growth**: The standard deviation of the lagged and adjusted traffic change rate during periods of rapid growth is {rapid_growth_std:.2f}%, indicating high volatility and suggesting that traffic responses are unpredictable during these times.")

        # Add Traffic per Page insights
        st.write("**Traffic per Page**: The average Traffic per Page metric gives you an indication of how efficiently each page contributes to overall traffic. Monitoring changes in this metric can provide insights into the overall quality and performance of your pages over time.")

        # Summary of findings
        st.write("### Summary of Findings")
        st.write(f"Based on these findings, it appears the twiddler algorithm rewards growth stability in the range of {stable_min:.2f}% to {stable_max:.2f}% with positive traffic changes after a lag of {lag_period} periods. However, if normalized page growth exceeds {rapid_growth_threshold:.2f}%, it is likely to reduce traffic by an average of {abs(rapid_growth_traffic):.2f}%, with a volatility of {rapid_growth_std:.2f}%, after the same lag. Additionally, monitoring 'Traffic per Page' can provide insights into the overall effectiveness and quality of your pages.")

        st.write("### Visualization")
        
        # Plotly visualization
        fig = go.Figure()

        # Page Growth Rate Line
        fig.add_trace(go.Scatter(
            x=df_ma[date_col],
            y=df_ma[f"Normalized Page Growth {window_size}MA"],
            mode='lines',
            name='Normalized Page Growth Rate (%)',
            line=dict(color='blue', width=2)
        ))

        # Lagged and Adjusted Traffic Change Rate Line
        fig.add_trace(go.Scatter(
            x=df_ma[date_col],
            y=df_ma[f"Lagged Adjusted Traffic Change {window_size}MA"],
            mode='lines',
            name='Lagged & Adjusted Traffic Change Rate (%)',
            line=dict(color='red', width=2)
        ))

        # Traffic per Page Line
        fig.add_trace(go.Scatter(
            x=df_ma[date_col],
            y=df_ma[f"Lagged Traffic per Page {window_size}MA"],
            mode='lines',
            name='Lagged Traffic per Page',
            line=dict(color='green', width=2, dash='dash')
        ))

        # Add zero line for clarity
        fig.add_shape(type="line",
                      x0=df_ma[date_col].min(), x1=df_ma[date_col].max(),
                      y0=0, y1=0,
                      line=dict(color="gray", width=1, dash="dash"))

        # Layout updates for a "cool and sexy" look
        fig.update_layout(
            title=f"{window_size}-Period Moving Average with {lag_period}-Period Lag",
            xaxis_title="Date",
            yaxis_title="Percentage (%)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(x=0, y=1.1, bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Analysis could not be completed due to insufficient data or a calculation error.")

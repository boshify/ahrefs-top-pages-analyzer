import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.title('Growth Rate Analyzer with Lagged Correlation and Traffic per Page')

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

    # Date range selector
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    # Filter data based on date range
    start_date, end_date = date_range
    df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

    # Aggregate data based on the selected date frame
    if date_frame == 'weekly':
        df = df.resample('W-Mon', on=date_col).sum().reset_index().sort_values(by=date_col)
    elif date_frame == 'monthly':
        df = df.resample('M', on=date_col).sum().reset_index().sort_values(by=date_col)

    # Calculate the actual number of pages added per period
    df['Pages Added'] = df[page_col].diff().fillna(0)

    # Calculate growth rates
    df['Page Growth Rate'] = df[page_col].pct_change() * 100
    df['Traffic Change Rate'] = df[traffic_col].pct_change() * 100

    # Calculate Traffic per Page
    df['Traffic per Page'] = df[traffic_col] / df[page_col]

    st.write("Calculated Growth Rates:")
    st.write(df[[date_col, 'Pages Added', 'Page Growth Rate', 'Traffic Change Rate', 'Traffic per Page']].dropna().head())

    # Allow user to select the lag period
    lag_period = st.slider("Select Lag Period (in periods)", min_value=0, max_value=12, value=1, step=1)

    if lag_period > 0:
        df['Lagged Traffic Change Rate'] = df['Traffic Change Rate'].shift(lag_period)
        df['Lagged Traffic per Page'] = df['Traffic per Page'].shift(lag_period)
    else:
        df['Lagged Traffic Change Rate'] = df['Traffic Change Rate']
        df['Lagged Traffic per Page'] = df['Traffic per Page']

    st.header("Analysis")
    
    def analyze_growth(df, window_size):
        if len(df) < window_size:
            st.warning(f"Not enough data to calculate a {window_size}-period moving average.")
            return None, None, None, None, None

        df[f"Page Growth {window_size}MA"] = df['Page Growth Rate'].rolling(window=window_size).mean()
        df[f"Lagged Traffic Change {window_size}MA"] = df['Lagged Traffic Change Rate'].rolling(window=window_size).mean()
        df[f"Lagged Traffic per Page {window_size}MA"] = df['Lagged Traffic per Page'].rolling(window=window_size).mean()
        df_ma = df.dropna(subset=[f"Page Growth {window_size}MA", f"Lagged Traffic Change {window_size}MA", f"Lagged Traffic per Page {window_size}MA"])

        if df_ma.empty:
            st.warning("Insufficient data after applying the moving average. Please try a shorter window size or adjust your data.")
            return None, None, None, None, None

        correlation_ma = df_ma[[f"Page Growth {window_size}MA", f"Lagged Traffic Change {window_size}MA"]].corr().iloc[0, 1]
        
        # Dynamically calculate stable and rapid growth thresholds
        stable_min = df_ma[f"Page Growth {window_size}MA"].quantile(0.25)
        stable_max = df_ma[f"Page Growth {window_size}MA"].quantile(0.75)
        rapid_growth_threshold = df_ma[f"Page Growth {window_size}MA"].quantile(0.90)

        # Calculating stable growth, rapid growth, and volatility during rapid growth
        stable_growth_mask = (df_ma[f"Page Growth {window_size}MA"] >= stable_min) & (df_ma[f"Page Growth {window_size}MA"] <= stable_max)
        rapid_growth_mask = df_ma[f"Page Growth {window_size}MA"] > rapid_growth_threshold

        stable_growth_traffic_change = df_ma[stable_growth_mask][f"Lagged Traffic Change {window_size}MA"].mean()
        rapid_growth_traffic_change = df_ma[rapid_growth_mask][f"Lagged Traffic Change {window_size}MA"].mean()
        rapid_growth_traffic_std = df_ma[rapid_growth_mask][f"Lagged Traffic Change {window_size}MA"].std()
        
        # Traffic per Page insights
        stable_growth_tpp = df_ma[stable_growth_mask][f"Lagged Traffic per Page {window_size}MA"].mean()
        rapid_growth_tpp = df_ma[rapid_growth_mask][f"Lagged Traffic per Page {window_size}MA"].mean()

        # Accurate calculation of pages added per period
        stable_pages_min = df_ma[stable_growth_mask]['Pages Added'].min() if not df_ma[stable_growth_mask].empty else 0
        stable_pages_max = df_ma[stable_growth_mask]['Pages Added'].max() if not df_ma[stable_growth_mask].empty else 0
        pages_per_period_stable = f"{stable_pages_min:.2f} to {stable_pages_max:.2f} pages added per {date_frame}"

        rapid_pages_min = df_ma[rapid_growth_mask]['Pages Added'].min() if not df_ma[rapid_growth_mask].empty else 0
        pages_per_period_rapid = f"more than {rapid_pages_min:.2f} pages added per {date_frame}"

        return correlation_ma, stable_growth_traffic_change, rapid_growth_traffic_change, rapid_growth_traffic_std, df_ma, stable_min, stable_max, rapid_growth_threshold, stable_growth_tpp, rapid_growth_tpp, pages_per_period_stable, pages_per_period_rapid

    # Allow user to select the moving average window size
    max_window_size = len(df)
    window_size = st.slider(f"Select Moving Average Window ({date_frame})", min_value=1, max_value=max_window_size, value=3, step=1)

    correlation, stable_growth_traffic, rapid_growth_traffic, rapid_growth_std, df_ma, stable_min, stable_max, rapid_growth_threshold, stable_growth_tpp, rapid_growth_tpp, pages_per_period_stable, pages_per_period_rapid = analyze_growth(df.copy(), window_size)
    
    if correlation is not None:
        st.subheader(f"{window_size}-Period Moving Average ({date_frame}):")
        st.write(f"Correlation (with {lag_period}-period lag): {correlation:.4f}")

        st.write("### Insights")
        if stable_growth_traffic is not None:
            st.write(f"**Stable Growth (between {stable_min:.2f}% and {stable_max:.2f}% or {pages_per_period_stable})**: During periods where page growth remains within this stable range, the average lagged traffic change rate is {stable_growth_traffic:.2f}%. This suggests that the algorithm rewards stable page growth, resulting in consistent {'increases' if stable_growth_traffic >= 0 else 'decreases'} in traffic after a lag of {lag_period} periods.")

        if rapid_growth_traffic is not None and not np.isnan(rapid_growth_traffic):
            delta_traffic = rapid_growth_traffic - stable_growth_traffic
            traffic_direction = "higher" if delta_traffic > 0 else "lower"
            st.write(f"**Rapid Growth (above {rapid_growth_threshold:.2f}% or {pages_per_period_rapid})**: When page growth exceeds this threshold, the average lagged traffic change rate is {rapid_growth_traffic:.2f}% ({abs(delta_traffic):.2f}% {traffic_direction} than stable). This indicates that rapid increases in page growth are associated with significant changes in traffic, potentially {'penalizing' if rapid_growth_traffic < stable_growth_traffic else 'rewarding'} sharp increases in page growth after a lag of {lag_period} periods.")

        if rapid_growth_std is not None and not np.isnan(rapid_growth_std):
            st.write(f"**Volatility during Rapid Growth**: The standard deviation of the lagged traffic change rate during periods of rapid growth is {rapid_growth_std:.2f}%, indicating high volatility and suggesting that traffic responses are unpredictable during these times.")

        # Add Traffic per Page insights
        if stable_growth_tpp is not None and rapid_growth_tpp is not None:
            tpp_change = rapid_growth_tpp - stable_growth_tpp
            tpp_percentage_change = (tpp_change / stable_growth_tpp) * 100 if stable_growth_tpp != 0 else 0
            tpp_direction = "higher" if tpp_change > 0 else "lower"
            st.write(f"**Traffic per Page during Stable Growth**: The average Traffic per Page during stable growth periods is {stable_growth_tpp:.2f}.")
            st.write(f"**Traffic per Page during Rapid Growth**: The average Traffic per Page during rapid growth periods is {rapid_growth_tpp:.2f}, which is {abs(tpp_change):.2f} ({abs(tpp_percentage_change):.2f}%) {tpp_direction} than during stable growth periods. This reflects how traffic efficiency changes during periods of rapid page growth.")

        # Summary of findings
        st.write("### Summary of Findings")
        tpp_summary = f"with a change in Traffic per Page of {abs(tpp_change):.2f} ({abs(tpp_percentage_change):.2f}%) {tpp_direction} from stable to rapid growth periods" if stable_growth_tpp is not None and rapid_growth_tpp is not None else ""
        st.write(f"Based on these findings, it appears the twiddler algorithm rewards growth stability in the range of {stable_min:.2f}% to {stable_max:.2f}% ({pages_per_period_stable}) with positive traffic changes after a lag of {lag_period} periods. However, if page growth exceeds {rapid_growth_threshold:.2f}% ({pages_per_period_rapid}), it is likely to {'reduce' if rapid_growth_traffic < stable_growth_traffic else 'increase'} traffic by an average of {abs(rapid_growth_traffic):.2f}%, with a volatility of {rapid_growth_std:.2f}%, after the same lag, {tpp_summary}.")

        st.write("### Visualization")
        
        # Plotly visualization
        fig = go.Figure()

        # Page Growth Rate Line
        fig.add_trace(go.Scatter(
            x=df_ma[date_col],
            y=df_ma[f"Page Growth {window_size}MA"],
            mode='lines',
            name='Page Growth Rate (%)',
            line=dict(color='blue', width=2)
        ))

        # Lagged and Adjusted Traffic Change Rate Line
        fig.add_trace(go.Scatter(
            x=df_ma[date_col],
            y=df_ma[f"Lagged Traffic Change {window_size}MA"],
            mode='lines',
            name='Lagged Traffic Change Rate (%)',
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

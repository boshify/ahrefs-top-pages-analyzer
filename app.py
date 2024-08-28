import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import timedelta

st.title('Growth Rate Analyzer with Ranking State Visualization')

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
    else:
        df = df.sort_values(by=date_col)

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

        # Identify Positive and Negative Ranking States
        df_ma['Ranking State'] = np.where(df_ma[f"Lagged Traffic per Page {window_size}MA"].diff() > 0, 'Positive', 'Negative')
        ranking_state_changes = df_ma[df_ma['Ranking State'] != df_ma['Ranking State'].shift(1)]

        ranking_state_report = []
        if not ranking_state_changes.empty:
            for i in range(len(ranking_state_changes) - 1):
                state = ranking_state_changes.iloc[i]['Ranking State']
                start_date = ranking_state_changes.iloc[i][date_col]
                end_date = ranking_state_changes.iloc[i + 1][date_col] - timedelta(days=1)
                avg_tpp_start = ranking_state_changes.iloc[i][f"Lagged Traffic per Page {window_size}MA"]
                avg_tpp_end = ranking_state_changes.iloc[i + 1][f"Lagged Traffic per Page {window_size}MA"]
                ranking_state_report.append(f"From {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, the site was in a {state} ranking state. The average traffic per page {'increased' if state == 'Positive' else 'decreased'} from {avg_tpp_start:.2f} to {avg_tpp_end:.2f}.")

            # Add the final state that runs until the end date
            final_state = ranking_state_changes.iloc[-1]['Ranking State']
            final_start_date = ranking_state_changes.iloc[-1][date_col]
            final_end_date = df_ma[date_col].max()
            final_avg_tpp_start = ranking_state_changes.iloc[-1][f"Lagged Traffic per Page {window_size}MA"]
            final_avg_tpp_end = df_ma.iloc[-1][f"Lagged Traffic per Page {window_size}MA"]
            ranking_state_report.append(f"From {final_start_date.strftime('%Y-%m-%d')} to {final_end_date.strftime('%Y-%m-%d')}, the site was in a {final_state} ranking state. The average traffic per page {'increased' if final_state == 'Positive' else 'decreased'} from {final_avg_tpp_start:.2f} to {final_avg_tpp_end:.2f}.")

        return correlation_ma, stable_growth_traffic_change, rapid_growth_traffic_change, rapid_growth_traffic_std, df_ma, stable_min, stable_max, rapid_growth_threshold, stable_growth_tpp, rapid_growth_tpp, ranking_state_report

    # Allow user to select the moving average window size
    max_window_size = len(df)
    window_size = st.slider(f"Select Moving Average Window ({date_frame})", min_value=1, max_value=max_window_size, value=3, step=1)

    correlation, stable_growth_traffic, rapid_growth_traffic, rapid_growth_std, df_ma, stable_min, stable_max, rapid_growth_threshold, stable_growth_tpp, rapid_growth_tpp, ranking_state_report = analyze_growth(df.copy(), window_size)
    
    if correlation is not None:
        st.subheader(f"{window_size}-Period Moving Average ({date_frame}):")
        st.write(f"Correlation (with {lag_period}-period lag): {correlation:.4f}")

        st.write("### Insights")
        if stable_growth_traffic is not None:
            st.write(f"**Stable Growth (between {stable_min:.2f}% and {stable_max:.2f}%)**: During periods where page growth remains within this stable range, the average lagged traffic change rate is {stable_growth_traffic:.2f}%. This suggests that the algorithm rewards stable page growth, resulting in consistent {'increases' if stable_growth_traffic >= 0 else 'decreases'} in traffic after a lag of {lag_period} periods.")

        if rapid_growth_traffic is not None and not np.isnan(rapid_growth_traffic):
            delta_traffic = rapid_growth_traffic - stable_growth_traffic
            traffic_direction = "higher" if delta_traffic > 0 else "lower"
            st.write(f"**Rapid Growth (above {rapid_growth_threshold:.2f}%)**: When page growth exceeds this threshold, the average lagged traffic change rate is {rapid_growth_traffic:.2f}% ({abs(delta_traffic):.2f}% {traffic_direction} than stable). This indicates that rapid increases in page growth are associated with significant changes in traffic, potentially {'penalizing' if rapid_growth_traffic < stable_growth_traffic else 'rewarding'} sharp increases in page growth after a lag of {lag_period} periods.")

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
        st.write(f"Based on these findings, it appears the twiddler algorithm rewards growth stability in the range of {stable_min:.2f}% to {stable_max:.2f}% with positive traffic changes after a lag of {lag_period} periods. However, if page growth exceeds {rapid_growth_threshold:.2f}%, it is likely to {'reduce' if rapid_growth_traffic < stable_growth_traffic else 'increase'} traffic by an average of {abs(rapid_growth_traffic):.2f}%, with a volatility of {rapid_growth_std:.2f}%, after the same lag, {tpp_summary}.")

        # Add ranking state report
        if ranking_state_report:
            st.write("### Ranking State Report")
            for report in ranking_state_report:
                st.write(report)

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

        # Add ranking state indicators
        for idx, row in df_ma.iterrows():
            if row['Ranking State'] == 'Positive':
                fig.add_vrect(
                    x0=row[date_col] - timedelta(days=1),
                    x1=row[date_col] + timedelta(days=1),
                    fillcolor="green", opacity=0.3, line_width=0
                )
            else:
                fig.add_vrect(
                    x0=row[date_col] - timedelta(days=1),
                    x1=row[date_col] + timedelta(days=1),
                    fillcolor="red", opacity=0.3, line_width=0
                )

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

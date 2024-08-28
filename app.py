import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import timedelta

st.title('Growth Rate Analyzer with Enhanced Ranking State Reports')

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

    # Calculate total pages added or removed per period
    df['Pages Added'] = df[page_col].diff().fillna(0)
    df['Page Change Rate'] = df['Pages Added'] / df[page_col].shift(1) * 100

    # Calculate Traffic per Page directly from the Traffic and Pages columns
    df['Traffic per Page'] = df[traffic_col] / df[page_col]

    # Calculate Traffic Change Rate compared to the previous period
    df['Traffic Change Rate'] = df[traffic_col].pct_change() * 100

    st.write("Calculated Metrics:")
    st.write(df[[date_col, 'Pages Added', 'Page Change Rate', 'Traffic Change Rate', 'Traffic per Page']].dropna().head())

    # Allow user to select the lag period
    lag_period = st.slider("Select Lag Period (in periods)", min_value=0, max_value=12, value=1, step=1)

    # Allow user to select the moving average window size
    max_window_size = len(df)
    window_size = st.slider(f"Select Moving Average Window ({date_frame})", min_value=1, max_value=max_window_size, value=3, step=1)

    # Apply the moving average to the relevant columns
    df[f"Page Change {window_size}MA"] = df['Page Change Rate'].rolling(window=window_size).mean()
    df[f"Traffic Change {window_size}MA"] = df['Traffic Change Rate'].rolling(window=window_size).mean()
    df[f"Traffic per Page {window_size}MA"] = df['Traffic per Page'].rolling(window=window_size).mean()

    if lag_period > 0:
        df[f"Lagged Traffic per Page {window_size}MA"] = df[f"Traffic per Page {window_size}MA"].shift(lag_period)
    else:
        df[f"Lagged Traffic per Page {window_size}MA"] = df[f"Traffic per Page {window_size}MA"]

    st.header("Ranking State Report")
    
    def generate_ranking_report(df):
        # Identify Positive and Negative Ranking States based on Traffic per Page change
        df['Ranking State'] = np.where(df[f"Lagged Traffic per Page {window_size}MA"].diff() > 0, 'Positive', 'Negative')
        ranking_state_changes = df[df['Ranking State'] != df['Ranking State'].shift(1)]

        ranking_state_report = []
        if not ranking_state_changes.empty:
            for i in range(len(ranking_state_changes) - 1):
                state = ranking_state_changes.iloc[i]['Ranking State']
                start_date = ranking_state_changes.iloc[i][date_col]
                end_date = ranking_state_changes.iloc[i + 1][date_col] - timedelta(days=1)
                avg_tpp_start = ranking_state_changes.iloc[i][f"Lagged Traffic per Page {window_size}MA"]
                avg_tpp_end = ranking_state_changes.iloc[i + 1][f"Lagged Traffic per Page {window_size}MA"]

                # Calculate page change and traffic change details
                page_change_total = (ranking_state_changes.iloc[i + 1][page_col] - ranking_state_changes.iloc[i][page_col]) / ranking_state_changes.iloc[i][page_col] * 100
                traffic_change_pct = ranking_state_changes.iloc[i + 1]['Traffic Change Rate']

                # Ensure ranking state is based on Traffic per Page change
                state = 'Positive' if avg_tpp_end > avg_tpp_start else 'Negative'

                ranking_state_report.append(
                    f"From {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, the site was in a **{state}** ranking state. "
                    f"The average traffic per page {'**increased**' if state == 'Positive' else '**decreased**'} from **{avg_tpp_start:.2f}** to **{avg_tpp_end:.2f}**. "
                    f"Pages {'**increased**' if page_change_total > 0 else '**decreased**'} by **{page_change_total:.2f}%** "
                    f"and traffic {'**increased**' if traffic_change_pct > 0 else '**decreased**'} by **{traffic_change_pct:.2f}%** compared to the previous period."
                )

            # Add the final state that runs until the end date
            final_state = 'Positive' if df.iloc[-1][f"Lagged Traffic per Page {window_size}MA"] > df.iloc[-2][f"Lagged Traffic per Page {window_size}MA"] else 'Negative'
            final_start_date = ranking_state_changes.iloc[-1][date_col]
            final_end_date = df[date_col].max()
            final_avg_tpp_start = ranking_state_changes.iloc[-1][f"Lagged Traffic per Page {window_size}MA"]
            final_avg_tpp_end = df.iloc[-1][f"Lagged Traffic per Page {window_size}MA"]

            # Calculate final period page change and traffic change
            final_page_change_total = (df.iloc[-1][page_col] - ranking_state_changes.iloc[-1][page_col]) / ranking_state_changes.iloc[-1][page_col] * 100
            final_traffic_change_pct = df.iloc[-1]['Traffic Change Rate']

            ranking_state_report.append(
                f"From {final_start_date.strftime('%Y-%m-%d')} to {final_end_date.strftime('%Y-%m-%d')}, the site was in a **{final_state}** ranking state. "
                f"The average traffic per page {'**increased**' if final_state == 'Positive' else '**decreased**'} from **{final_avg_tpp_start:.2f}** to **{final_avg_tpp_end:.2f}**. "
                f"Pages {'**increased**' if final_page_change_total > 0 else '**decreased**'} by **{final_page_change_total:.2f}%** "
                f"and traffic {'**increased**' if final_traffic_change_pct > 0 else '**decreased**'} by **{final_traffic_change_pct:.2f}%** compared to the previous period."
            )

        return df, ranking_state_report

    df, ranking_state_report = generate_ranking_report(df.copy())

    if ranking_state_report:
        for report in ranking_state_report:
            st.write(report)

    st.write("### Visualization")
    
    # Plotly visualization
    fig = go.Figure()

    # Page Growth Rate Line (right y-axis)
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[f"Page Change {window_size}MA"],
        mode='lines',
        name='Page Change Rate (%)',
        line=dict(color='blue', width=2),
        yaxis="y2"
    ))

    # Lagged Traffic Change Rate Line (right y-axis)
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[f"Traffic Change {window_size}MA"],
        mode='lines',
        name='Traffic Change Rate (%)',
        line=dict(color='red', width=2),
        yaxis="y2"
    ))

    # Traffic per Page Line (left y-axis)
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[f"Lagged Traffic per Page {window_size}MA"],
        mode='lines',
        name='Traffic per Page',
        line=dict(color='green', width=2, dash='dash')
    ))

    # Add ranking state indicators to match the ranking state report
    for idx, row in df.iterrows():
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

    # Add zero line for clarity on the right y-axis
    fig.add_shape(type="line",
                  x0=df[date_col].min(), x1=df[date_col].max(),
                  y0=0, y1=0,
                  yref="y2",
                  line=dict(color="gray", width=1, dash="dash"))

    # Layout updates for a clear and appealing look
    fig.update_layout(
        title=f"{date_frame.capitalize()} Ranking State Visualization",
        xaxis_title="Date",
        yaxis_title="Traffic per Page",
        yaxis2=dict(title="Percentage (%)", overlaying="y", side="right"),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(x=0, y=1.1, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=40, r=40, t=40, b=40),
        height=600  # Make the chart bigger
    )

    # Add scroll and zoom functionality
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig, use_container_width=True)

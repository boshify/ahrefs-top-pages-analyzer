import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import timedelta

st.set_page_config(layout="wide")
st.title('Ahrefs Top Pages Analyzer')

@st.cache_data(show_spinner=False)
def calculate_moving_averages(df, page_col, traffic_col, window_size):
    df[f"Page Change {window_size}MA"] = df['Page Change Rate'].rolling(window=window_size).mean()
    df[f"Traffic Change {window_size}MA"] = df['Traffic Change Rate'].rolling(window=window_size).mean()
    df[f"Traffic per Page {window_size}MA"] = df['Traffic per Page'].rolling(window=window_size).mean()
    return df

def generate_ranking_report_table(df, window_size, date_col, page_col, traffic_col):
    df['Ranking State'] = np.where(df[f"Traffic per Page {window_size}MA"].diff().fillna(0) > 0, 'Positive', 'Negative')
    ranking_state_changes = df[df['Ranking State'] != df['Ranking State'].shift(1)].copy()

    report_rows = []
    if not ranking_state_changes.empty:
        for i in range(len(ranking_state_changes) - 1):
            state = ranking_state_changes.iloc[i]['Ranking State']
            start_date = ranking_state_changes.iloc[i][date_col]
            end_date = ranking_state_changes.iloc[i + 1][date_col] - timedelta(days=1)
            avg_tpp_start = ranking_state_changes.iloc[i][f"Traffic per Page {window_size}MA"]
            avg_tpp_end = ranking_state_changes.iloc[i + 1][f"Traffic per Page {window_size}MA"]

            indexed_pages_before = ranking_state_changes.iloc[i][page_col]
            indexed_pages_after = ranking_state_changes.iloc[i + 1][page_col]
            indexed_pages_change_pct = (indexed_pages_after - indexed_pages_before) / indexed_pages_before * 100 if indexed_pages_before else np.nan

            total_traffic_before = ranking_state_changes.iloc[i][traffic_col]
            total_traffic_after = ranking_state_changes.iloc[i + 1][traffic_col]
            total_traffic_change_pct = (total_traffic_after - total_traffic_before) / total_traffic_before * 100 if total_traffic_before else np.nan

            report_rows.append({
                "Date Start": start_date.strftime('%Y-%m-%d'),
                "Date End": end_date.strftime('%Y-%m-%d'),
                "Ranking State": state,
                "Avg. Traffic Per Page Before": avg_tpp_start,
                "Avg. Traffic Per Page After": avg_tpp_end,
                "Indexed Pages Before": indexed_pages_before,
                "Indexed Pages After": indexed_pages_after,
                "Indexed Pages Change %": indexed_pages_change_pct,
                "Total Traffic Before": total_traffic_before,
                "Total Traffic After": total_traffic_after,
                "Total Traffic Change %": total_traffic_change_pct,
            })

        # Final state until end date
        final_state = df.iloc[-1]['Ranking State']
        final_start_date = ranking_state_changes.iloc[-1][date_col]
        final_end_date = df[date_col].max()
        final_avg_tpp_start = ranking_state_changes.iloc[-1][f"Traffic per Page {window_size}MA"]
        final_avg_tpp_end = df.iloc[-1][f"Traffic per Page {window_size}MA"]
        final_indexed_pages_before = ranking_state_changes.iloc[-1][page_col]
        final_indexed_pages_after = df.iloc[-1][page_col]
        final_indexed_pages_change_pct = (final_indexed_pages_after - final_indexed_pages_before) / final_indexed_pages_before * 100 if final_indexed_pages_before else np.nan
        final_total_traffic_before = ranking_state_changes.iloc[-1][traffic_col]
        final_total_traffic_after = df.iloc[-1][traffic_col]
        final_total_traffic_change_pct = (final_total_traffic_after - final_total_traffic_before) / final_total_traffic_before * 100 if final_total_traffic_before else np.nan

        report_rows.append({
            "Date Start": final_start_date.strftime('%Y-%m-%d'),
            "Date End": final_end_date.strftime('%Y-%m-%d'),
            "Ranking State": final_state,
            "Avg. Traffic Per Page Before": final_avg_tpp_start,
            "Avg. Traffic Per Page After": final_avg_tpp_end,
            "Indexed Pages Before": final_indexed_pages_before,
            "Indexed Pages After": final_indexed_pages_after,
            "Indexed Pages Change %": final_indexed_pages_change_pct,
            "Total Traffic Before": final_total_traffic_before,
            "Total Traffic After": final_total_traffic_after,
            "Total Traffic Change %": final_total_traffic_change_pct,
        })

    report_df = pd.DataFrame(report_rows)
    float_cols = ["Avg. Traffic Per Page Before", "Avg. Traffic Per Page After", "Indexed Pages Change %", "Total Traffic Change %"]
    report_df[float_cols] = report_df[float_cols].round(2)
    return report_df

# Sidebar Inputs
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    try:
        if uploaded_file is not None:
            if 'df' not in st.session_state:
                df = pd.read_csv(uploaded_file)
                st.session_state['df'] = df
            else:
                df = st.session_state['df']

            st.write("Data Preview:")
            st.write(df.head())

            # Default to "Date", "Pages", "Traffic" if present
            columns = list(df.columns)
            def default_column(target):
                for col in columns:
                    if col.lower() == target.lower():
                        return col
                return columns[0]

            date_col = st.selectbox("Select the column for 'Date':", columns, index=columns.index(default_column('Date')) if 'Date' in columns else 0)
            page_col = st.selectbox("Select the column for 'Pages':", columns, index=columns.index(default_column('Pages')) if 'Pages' in columns else 0)
            traffic_col = st.selectbox("Select the column for 'Traffic':", columns, index=columns.index(default_column('Traffic')) if 'Traffic' in columns else 0)

            if date_col and page_col and traffic_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
                df = df.dropna(subset=[date_col])

                date_frame = st.selectbox("Select Date Frame:", ['daily', 'weekly', 'monthly'])
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
                start_date, end_date = date_range
                df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

                if date_frame == 'weekly':
                    df = df.resample('W-Mon', on=date_col).sum().reset_index().sort_values(by=date_col)
                elif date_frame == 'monthly':
                    df = df.resample('M', on=date_col).sum().reset_index().sort_values(by=date_col)
                else:
                    df = df.sort_values(by=date_col)

                df['Pages Added'] = df[page_col].diff().fillna(0)
                df['Page Change Rate'] = df['Pages Added'] / (df[page_col].shift(1) + df['Pages Added']).replace({0: np.nan}) * 100
                df['Traffic per Page'] = df[traffic_col] / df[page_col]
                df['Traffic Change Rate'] = df[traffic_col].pct_change() * 100

                # 1-52 weeks for MA, default = 1
                max_window_size = min(len(df), 52)
                window_size = st.slider(f"Select Moving Average Window ({date_frame})", min_value=1, max_value=max_window_size, value=1, step=1)

                df = calculate_moving_averages(df, page_col, traffic_col, window_size)
                df['Ranking State'] = np.where(df[f"Traffic per Page {window_size}MA"].diff().fillna(0) > 0, 'Positive', 'Negative')

                # Weighted averages for summary (no change here)
                positive_weighted_avg = np.average(df[df['Ranking State'] == 'Positive'][f"Page Change {window_size}MA"].dropna())
                negative_weighted_avg = np.average(df[df['Ranking State'] == 'Negative'][f"Page Change {window_size}MA"].dropna())
                summary_report = f"""
                **Summary Report:**
                - **Page Increase Threshold for Positive Ranking States (Weighted Average):** {positive_weighted_avg:.2f}%
                - **Page Increase Threshold for Negative Ranking States (Weighted Average):** {negative_weighted_avg:.2f}%
                """

    except Exception:
        st.session_state['input_error'] = True
    else:
        st.session_state['input_error'] = False

# Main Panel Logic
if uploaded_file is not None and not st.session_state.get('input_error', False):
    st.write("### Visualization")

    # Prepare hover text including Total Pages for each point
    pages_tooltip = df[page_col].astype(int).astype(str)
    date_tooltip = df[date_col].dt.strftime('%Y-%m-%d')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[f"Page Change {window_size}MA"],
        mode='lines',
        name='Page Change Rate (%)',
        line=dict(color='#3288d7', width=3),
        yaxis="y2",
        hovertemplate=(
            'Date: %{x}<br>'
            'Page Change Rate: %{y:.2f}%<br>'
            f'Total Pages: %{customdata[0]}'
            '<extra></extra>'
        ),
        customdata=np.stack([pages_tooltip], axis=-1),
    ))
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[f"Traffic Change {window_size}MA"],
        mode='lines',
        name='Traffic Change Rate (%)',
        line=dict(color='#ff8800', width=3),
        yaxis="y2",
        hovertemplate=(
            'Date: %{x}<br>'
            'Traffic Change Rate: %{y:.2f}%<br>'
            f'Total Pages: %{customdata[0]}'
            '<extra></extra>'
        ),
        customdata=np.stack([pages_tooltip], axis=-1),
    ))
    fig.add_shape(type="line",
                  x0=df[date_col].min(), x1=df[date_col].max(),
                  y0=0, y1=0, yref="y2", line=dict(color="gray", width=2, dash="dash"))
    fig.update_layout(
        title=f"{date_frame.capitalize()} Ranking State Visualization",
        xaxis_title="Date",
        yaxis=dict(title="Traffic per Page", side="left"),
        yaxis2=dict(title="Percentage (%)", side="right", overlaying="y", showgrid=False, range=[-50, 50], type='linear'),
        template="plotly_dark", hovermode="x unified",
        legend=dict(x=0, y=1.1, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=20, r=20, t=120, b=100),
        height=1000
    )
    fig.update_xaxes(rangeslider_visible=True)
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[f"Traffic per Page {window_size}MA"],
        mode='lines',
        name='Traffic per Page',
        line=dict(color='green', width=4, dash='dash'),
        yaxis="y",
        hovertemplate=(
            'Date: %{x}<br>'
            'Traffic per Page: %{y:.2f}<br>'
            f'Total Pages: %{customdata[0]}'
            '<extra></extra>'
        ),
        customdata=np.stack([pages_tooltip], axis=-1),
    ))

    for idx, row in df.iterrows():
        if row['Ranking State'] == 'Positive':
            fig.add_vrect(x0=row[date_col] - timedelta(days=1), x1=row[date_col] + timedelta(days=1),
                          fillcolor="green", opacity=0.2, line_width=0)
        elif row['Ranking State'] == 'Negative':
            fig.add_vrect(x0=row[date_col] - timedelta(days=1), x1=row[date_col] + timedelta(days=1),
                          fillcolor="red", opacity=0.2, line_width=0)

    st.plotly_chart(fig, use_container_width=True)
    st.header("Ranking State Report (Table)")
    st.write(summary_report)

    # Display Table Instead of Verbal Report
    ranking_report_df = generate_ranking_report_table(df.copy(), window_size, date_col, page_col, traffic_col)
    if not ranking_report_df.empty:
        st.dataframe(ranking_report_df)
    else:
        st.write("No ranking state periods found for the current filters.")

elif uploaded_file is not None and st.session_state.get('input_error', False):
    st.error("Check Your Inputs")
else:
    st.write("Please add Date, Pages, and Traffic fields to begin the analysis.")

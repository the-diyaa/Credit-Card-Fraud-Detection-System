
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================
# 1. PROJECT PATHS AND IMPORTS
# =====================================================

# app.py is inside the app folder.
# The project root is one directory above it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.predict import predict_transactions
except ModuleNotFoundError as exc:
    st.error(
        "Unable to import the prediction module. "
        "Please check that src/predict.py exists."
    )
    st.exception(exc)
    st.stop()

# =====================================================
# 2. STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FraudGuard | Smart Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PROBABILITY_COLUMN = "Fraud Probability (%)"

# =====================================================
# 3. VIBRANT DASHBOARD DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #F5F3FF,
        #EFF6FF,
        #F0FDFA
    );
    color: #172554;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
    display: none;
}

#MainMenu,
footer {
    visibility: hidden;
}

h1, h2, h3 {
    color: #172554 !important;
    font-weight: 750 !important;
}

/* Hero Header */

.hero {
    background: linear-gradient(
        120deg,
        #312E81,
        #4F46E5,
        #0891B2
    );
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(49, 46, 129, 0.20);
}

.hero h1 {
    color: white !important;
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    color: #E0E7FF;
    font-size: 16px;
    margin: 0;
}

/* Upload Area */

[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #818CF8;
    border-radius: 16px;
    padding: 18px;
}

/* Metric Cards */

[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #E0E7FF;
    box-shadow: 0 5px 15px rgba(79, 70, 229, 0.07);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #312E81 !important;
    font-weight: 800;
}

/* Buttons */

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(
        90deg,
        #4F46E5,
        #0891B2
    );
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 20px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(
        90deg,
        #4338CA,
        #0E7490
    );
    color: white;
    border: none;
}

/* Footer */

.footer {
    text-align: center;
    color: #64748B;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. HELPER FUNCTION
# =====================================================

def prepare_prediction_output(predictions, row_count):

    if isinstance(predictions, pd.Series):
        prediction_df = predictions.to_frame()

    elif isinstance(predictions, pd.DataFrame):
        prediction_df = predictions.copy()

    else:
        prediction_df = pd.DataFrame(predictions)

    if len(prediction_df) != row_count:
        raise ValueError(
            "Prediction count does not match uploaded transactions."
        )

    possible_columns = [
        PROBABILITY_COLUMN,
        "fraud_probability",
        "Fraud Probability",
        "probability",
        "Probability",
        "fraud_prob",
        "Fraud_Probability"
    ]

    probability_column = next(
        (
            col for col in possible_columns
            if col in prediction_df.columns
        ),
        None
    )

    if probability_column is None:

        if prediction_df.shape[1] == 1:
            probability_column = prediction_df.columns[0]

        else:
            raise ValueError(
                "Fraud probability column not found in model output. "
                "Check your src/predict.py file."
            )

    probabilities = pd.to_numeric(
        prediction_df[probability_column],
        errors="coerce"
    )

    if probabilities.isna().any():
        raise ValueError(
            "Model returned invalid fraud probability values."
        )

    # Convert probability values from 0-1 to 0-100.
    if (
        not probabilities.empty
        and probabilities.min() >= 0
        and probabilities.max() <= 1
    ):
        probabilities = probabilities * 100

    if ((probabilities < 0) | (probabilities > 100)).any():
        raise ValueError(
            "Fraud probabilities must be between 0 and 100."
        )

    prediction_df = prediction_df.drop(
        columns=[probability_column],
        errors="ignore"
    )

    prediction_df[PROBABILITY_COLUMN] = probabilities.to_numpy()

    return prediction_df


# =====================================================
# 5. HERO HEADER
# =====================================================

st.markdown(
    """<div class="hero">
<h1>💳 FraudGuard</h1>
<p>Intelligent Credit Card Fraud Detection and Risk Analysis Dashboard</p>
</div>""",
    unsafe_allow_html=True
)



# =====================================================
# 6. DATASET UPLOAD
# =====================================================

st.markdown("## 📂 Upload Transaction Dataset")

st.write(
    "Upload your credit card transaction CSV file "
    "to detect potentially fraudulent transactions."
)

uploaded_file = st.file_uploader(
    "Choose your CSV file",
    type=["csv"],
    key="fraud_csv_upload"
)

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV file is empty.")
            st.stop()

        st.success(
            f"Successfully uploaded: {uploaded_file.name}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Dataset Rows",
            f"{len(df):,}"
        )

        col2.metric(
            "Dataset Columns",
            len(df.columns)
        )

        col3.metric(
            "File Size",
            f"{uploaded_file.size / 1024:.1f} KB"
        )

        with st.expander("Preview Uploaded Dataset"):

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

        # Remove actual target column if present.
        prediction_data = df.drop(
            columns=["Class"],
            errors="ignore"
        )

        st.divider()

        # =====================================================
        # 7. ANALYZE TRANSACTIONS
        # =====================================================

        if st.button(
            "🚀 Analyze Transactions",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Analyzing transactions using the trained model..."
                ):

                    if prediction_data.empty:
                        raise ValueError(
                            "No feature columns found in the dataset."
                        )

                    if prediction_data.isnull().any().any():

                        missing_columns = prediction_data.columns[
                            prediction_data.isnull().any()
                        ].tolist()

                        raise ValueError(
                            f"Missing values found in: {missing_columns}"
                        )

                    non_numeric = prediction_data.select_dtypes(
                        exclude="number"
                    ).columns.tolist()

                    if non_numeric:
                        raise ValueError(
                            f"Non-numeric columns found: {non_numeric}"
                        )

                    # Run trained machine learning model.
                    predictions = predict_transactions(
                        prediction_data
                    )

                    prediction_output = prepare_prediction_output(
                        predictions,
                        len(prediction_data)
                    )

                    # Combine original data and predictions.
                    output = pd.concat(
                        [
                            prediction_data.reset_index(drop=True),
                            prediction_output.reset_index(drop=True)
                        ],
                        axis=1
                    )

                    # Save results in session state.
                    st.session_state["fraud_results"] = output

                    st.session_state["fraud_filename"] = (
                        uploaded_file.name
                    )

                st.success(
                    "Transaction analysis completed successfully!"
                )

            except Exception as e:

                st.error(f"Analysis failed: {e}")

    except pd.errors.EmptyDataError:

        st.error(
            "The uploaded CSV file does not contain readable data."
        )

    except UnicodeDecodeError:

        st.error(
            "Unable to read this CSV encoding. "
            "Please save the file as UTF-8 CSV."
        )

    except Exception as e:

        st.error(f"Unable to read the uploaded CSV: {e}")


# =====================================================
# 8. DISPLAY DASHBOARD RESULTS
# =====================================================

if "fraud_results" in st.session_state:

    results = st.session_state["fraud_results"].copy()

    if PROBABILITY_COLUMN not in results.columns:
        st.error(
            "Fraud probability column is missing. "
            "Please analyze the dataset again."
        )
        st.stop()

    st.divider()

    st.markdown("## 📊 Fraud Detection Dashboard")

    st.caption(
        f"Analysis Report: "
        f"{st.session_state.get('fraud_filename', 'Uploaded Dataset')}"
    )

    # =====================================================
    # 9. DETECTION THRESHOLD
    # =====================================================

    st.markdown("### ⚙️ Detection Settings")

    threshold_percent = st.slider(
        "Fraud Detection Threshold (%)",
        min_value=1,
        max_value=99,
        value=50,
        step=1,
        help=(
            "Adjust the probability required to classify "
            "a transaction as fraudulent."
        )
    )

    # Classify transactions using selected threshold.
    results["Prediction"] = (
        results[PROBABILITY_COLUMN] >= threshold_percent
    ).astype(int)

    results["Status"] = results["Prediction"].map({
        0: "Genuine",
        1: "Fraudulent"
    })

    # Risk classification.
    results["Risk Level"] = pd.cut(
        results[PROBABILITY_COLUMN],
        bins=[-0.001, 30, 70, 100],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    # =====================================================
    # 10. KPI CARDS
    # =====================================================

    total = len(results)

    fraud = int(
        (results["Prediction"] == 1).sum()
    )

    genuine = int(
        (results["Prediction"] == 0).sum()
    )

    fraud_rate = (
        fraud / total * 100 if total else 0
    )

    average_score = (
        results[PROBABILITY_COLUMN].mean()
        if total else 0
    )

    st.markdown("### 📈 Transaction Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{total:,}"
    )

    c2.metric(
        "Genuine Transactions",
        f"{genuine:,}"
    )

    c3.metric(
        "Fraudulent Transactions",
        f"{fraud:,}",
        delta=f"{fraud_rate:.2f}% of total",
        delta_color="inverse"
    )

    c4.metric(
        "Average Fraud Score",
        f"{average_score:.2f}%"
    )

    st.divider()

    # =====================================================
    # 11. INTERACTIVE ANALYTICS
    # =====================================================

    st.markdown("## 📉 Interactive Analytics")

    chart1, chart2 = st.columns(2)

    # DONUT CHART

    with chart1:

        st.markdown("### Transaction Distribution")

        distribution = pd.DataFrame({
            "Type": ["Genuine", "Fraudulent"],
            "Count": [genuine, fraud]
        })

        fig1 = px.pie(
            distribution,
            names="Type",
            values="Count",
            hole=0.62,
            color="Type",
            color_discrete_map={
                "Genuine": "#4F46E5",
                "Fraudulent": "#F43F5E"
            }
        )

        fig1.update_traces(
            textinfo="label+percent",
            hovertemplate=(
                "%{label}<br>"
                "Count: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            )
        )

        fig1.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#172554",
            legend_title_text="",
            margin=dict(t=25, b=25, l=10, r=10)
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    # FRAUD PROBABILITY HISTOGRAM

    with chart2:

        st.markdown("### Fraud Probability Distribution")

        fig2 = px.histogram(
            results,
            x=PROBABILITY_COLUMN,
            nbins=30,
            color_discrete_sequence=["#06B6D4"],
            labels={
                PROBABILITY_COLUMN: "Fraud Probability (%)"
            }
        )

        fig2.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#172554",
            xaxis_title="Estimated Fraud Probability",
            yaxis_title="Number of Transactions"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # =====================================================
    # 12. TRANSACTION TREND
    # =====================================================

    st.markdown("### ⏱️ Transaction Analysis Over Time")

    if "Time" in results.columns:

        time_data = results.copy()

        time_data["Time Group"] = (
            pd.to_numeric(
                time_data["Time"],
                errors="coerce"
            ) // 3600
        )

        time_data = time_data.dropna(
            subset=["Time Group"]
        )

        time_data["Time Group"] = (
            time_data["Time Group"].astype(int)
        )

        if not time_data.empty:

            hourly_data = (
                time_data.groupby(
                    ["Time Group", "Status"],
                    observed=True
                )
                .size()
                .reset_index(name="Transaction Count")
            )

            fig3 = px.line(
                hourly_data,
                x="Time Group",
                y="Transaction Count",
                color="Status",
                markers=True,
                color_discrete_map={
                    "Genuine": "#4F46E5",
                    "Fraudulent": "#F43F5E"
                },
                labels={
                    "Time Group": "Hours Since First Recorded Transaction",
                    "Transaction Count": "Number of Transactions"
                }
            )

            fig3.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#172554",
                hovermode="x unified"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

        else:
            st.info(
                "No usable values were found in the Time column."
            )

    else:
        st.info(
            "A Time column was not found. "
            "The transaction trend chart is unavailable."
        )

    # =====================================================
    # 13. RISK LEVEL AND STATUS CHARTS
    # =====================================================

    chart3, chart4 = st.columns(2)

    # RISK LEVEL CHART

    with chart3:

        st.markdown("### Risk Level Distribution")

        risk_counts = (
            results["Risk Level"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High"],
                fill_value=0
            )
            .rename_axis("Risk Level")
            .reset_index(name="Count")
        )

        fig4 = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            text="Count",
            color_discrete_map={
                "Low": "#10B981",
                "Medium": "#F59E0B",
                "High": "#F43F5E"
            }
        )

        fig4.update_layout(
            height=350,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#172554"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    # TRANSACTION STATUS CHART

    with chart4:

        st.markdown("### Transaction Status")

        status_counts = (
            results["Status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Count")
        )

        fig5 = px.bar(
            status_counts,
            x="Status",
            y="Count",
            color="Status",
            text="Count",
            color_discrete_map={
                "Genuine": "#4F46E5",
                "Fraudulent": "#F43F5E"
            }
        )

        fig5.update_layout(
            height=350,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#172554"
        )

        st.plotly_chart(
            fig5,
            use_container_width=True
        )

    # =====================================================
    # 14. TRANSACTION EXPLORER
    # =====================================================

    st.divider()

    st.markdown("## 🔎 Transaction Explorer")

    filter1, filter2 = st.columns(2)

    with filter1:

        selected_status = st.selectbox(
            "Filter by Transaction Status",
            ["All", "Genuine", "Fraudulent"]
        )

    with filter2:

        selected_risk = st.selectbox(
            "Filter by Risk Level",
            ["All", "Low", "Medium", "High"]
        )

    filtered_results = results.copy()

    if selected_status != "All":

        filtered_results = filtered_results[
            filtered_results["Status"] == selected_status
        ]

    if selected_risk != "All":

        filtered_results = filtered_results[
            filtered_results["Risk Level"] == selected_risk
        ]

    st.write(
        f"Showing {len(filtered_results):,} matching transactions"
    )

    st.dataframe(
        filtered_results,
        use_container_width=True,
        height=400
    )

    # =====================================================
    # 15. DOWNLOAD REPORT
    # =====================================================

    st.divider()

    st.markdown("## 📥 Download Reports")

    csv_data = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇ Download Complete Fraud Analysis Report",
        data=csv_data,
        file_name="fraud_analysis_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    # CLEAR DASHBOARD

    if st.button(
        "🗑 Clear Dashboard Results",
        use_container_width=True
    ):

        st.session_state.pop("fraud_results", None)
        st.session_state.pop("fraud_filename", None)

        st.rerun()


# =====================================================
# 16. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align: center; padding: 15px 0;">
        <h4 style="color: #4F46E5; margin-bottom: 8px;">
            💳 FraudGuard
        </h4>
        <p style="color: #64748B; margin-bottom: 5px;">
            Credit Card Fraud Detection and Risk Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


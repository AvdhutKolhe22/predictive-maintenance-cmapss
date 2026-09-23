
import streamlit as st
import pandas as pd

from app import analyze_engine, DATA


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)


# -------------------------------------------------
# CUSTOM STYLING
# -------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #9CA3AF;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 600;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown(
    '<div class="main-title">Predictive Maintenance Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">NASA C-MAPSS FD001 | CNN-BiLSTM Multi-Task Model</div>',
    unsafe_allow_html=True
)

st.divider()


# -------------------------------------------------
# CHECK DATA
# -------------------------------------------------

if DATA is None:

    st.error(
        "Dataset could not be loaded. "
        "Check the data/raw/test_FD001.txt file."
    )

    st.stop()


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header("Engine Selection")

engine_ids = sorted(
    DATA["unit_id"].unique().astype(int).tolist()
)

selected_engine = st.sidebar.selectbox(
    "Select Engine",
    engine_ids
)

analyze_button = st.sidebar.button(
    "Analyze Engine",
    type="primary"
)


# -------------------------------------------------
# DEFAULT INSTRUCTIONS
# -------------------------------------------------

if not analyze_button:

    st.info(
        "Select an engine from the sidebar "
        "and click Analyze Engine."
    )

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Engines",
            DATA["unit_id"].nunique()
        )

    with col2:
        st.metric(
            "Total Records",
            f"{len(DATA):,}"
        )

    with col3:
        st.metric(
            "Window Size",
            "30 cycles"
        )

    st.stop()


# -------------------------------------------------
# ENGINE ANALYSIS
# -------------------------------------------------

try:

    with st.spinner("Analyzing engine..."):

        result = analyze_engine(
            selected_engine
        )

except Exception as exc:

    st.error(
        f"Prediction failed: {exc}"
    )

    st.stop()


# -------------------------------------------------
# EXTRACT RESULTS
# -------------------------------------------------

predicted_rul = result["predicted_rul"]

failure_probability = result[
    "failure_probability"
]

status = result["status"]

recommendation = result[
    "recommendation"
]

observed_cycles = result[
    "observed_cycles"
]

latest_cycle = result[
    "latest_cycle"
]


# -------------------------------------------------
# STATUS COLOR
# -------------------------------------------------

if status == "CRITICAL":

    status_color = "#DC2626"

elif status == "WARNING":

    status_color = "#F59E0B"

else:

    status_color = "#16A34A"


# -------------------------------------------------
# ENGINE SUMMARY
# -------------------------------------------------

st.subheader(
    f"Engine {selected_engine} Analysis"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Predicted RUL",
        f"{predicted_rul:.2f} cycles"
    )

with col2:

    st.metric(
        "Failure Risk",
        f"{failure_probability:.2f}%"
    )

with col3:

    st.metric(
        "Observed Cycles",
        observed_cycles
    )

with col4:

    st.metric(
        "Latest Cycle",
        latest_cycle
    )


st.markdown("### Maintenance Status")

st.markdown(
    f"""
    <div class="status-box"
         style="background-color: {status_color};
                color: white;">
        {status}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("### Recommendation")

if status == "CRITICAL":

    st.error(recommendation)

elif status == "WARNING":

    st.warning(recommendation)

else:

    st.success(recommendation)


# -------------------------------------------------
# PREDICTION TRAJECTORY
# -------------------------------------------------

st.subheader("Prediction Trajectory")

trajectory = pd.DataFrame(
    result["trajectory"]
)

if not trajectory.empty:

    trajectory = trajectory.set_index(
        "cycle"
    )

    st.line_chart(
        trajectory[
            ["rul", "failure_probability"]
        ]
    )


# -------------------------------------------------
# DETAILED PREDICTIONS
# -------------------------------------------------

st.subheader("Detailed Prediction History")

display_columns = [
    "cycle",
    "rul",
    "failure_probability",
    "status"
]

st.dataframe(
    pd.DataFrame(
        result["trajectory"]
    )[display_columns],
    use_container_width=True,
    hide_index=True
)


# -------------------------------------------------
# WINDOW INFORMATION
# -------------------------------------------------

st.subheader("Latest Analysis Window")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Window Start",
        result["window_start"]
    )

with col2:

    st.metric(
        "Window End",
        result["window_end"]
    )

with col3:

    st.metric(
        "Window Length",
        "30 cycles"
    )


st.caption(
    "Predictions are generated using the trained "
    "CNN-BiLSTM multi-task models."
)
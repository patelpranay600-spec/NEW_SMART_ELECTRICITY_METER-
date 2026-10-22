# dashboard.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.webhooks import AnomalyWebhookDispatcher

st.set_page_config(page_title="Smart Meter XAI Neural Network SOC", layout="wide")
st.title("⚡ Smart Meter Anomaly Detection — Explainable Neural Network (XAI) SOC")

@st.cache_resource
def load_detector():
    return joblib.load("artifacts/explainable_nn_detector.joblib")

detector = load_detector()

# Sidebar Controls
st.sidebar.header("⚙️ Neural Network Operating Controls")
operating_preset = st.sidebar.selectbox(
    "Operating Mode Preset",
    ["Balanced SOC Mode (T=0.35)", "High-Sensitivity Mode (T=0.25)", "Zero-False-Alarm Mode (T=0.50)", "Custom Threshold"]
)
preset_map = {
    "Balanced SOC Mode (T=0.35)": 0.35,
    "High-Sensitivity Mode (T=0.25)": 0.25,
    "Zero-False-Alarm Mode (T=0.50)": 0.50,
}
default_t = preset_map.get(operating_preset, 0.35)
threshold = st.sidebar.slider("Probability Threshold (T)", 0.10, 0.90, default_t, 0.01)
detector.threshold = threshold

st.sidebar.markdown("---")
st.sidebar.header("🔔 Automated Webhook Config")
webhook_url = st.sidebar.text_input("SOC Webhook URL (Slack / Teams / PagerDuty)", value="")
min_sev = st.sidebar.selectbox("Minimum Webhook Severity", ["CRITICAL", "HIGH", "MEDIUM-ALERT"], index=1)
dry_run_mode = st.sidebar.checkbox("Dry-Run Mode (Log payloads locally)", value=True)

# CSV Ingestion
uploaded_file = st.file_uploader("📂 Upload Smart Meter Telemetry CSV", type=["csv"])
input_path = "data/test_smart_meter.csv"
if uploaded_file is not None:
    input_path = "data/uploaded_batch.csv"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

results_df = detector.explain_and_predict_csv(input_path, "data/explained_predictions.csv")
flagged_df = results_df[results_df["Predicted_Label"] == "Abnormal"]

# Top KPI Row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Records Scored", f"{len(results_df):,}")
c2.metric("Flagged Anomalies", f"{len(flagged_df):,}", f"{len(flagged_df)/len(results_df)*100:.1f}%")
c3.metric("CRITICAL (P ≥ 0.80)", f"{int((results_df['Severity_Tier'] == 'CRITICAL').sum())}")
c4.metric("HIGH (0.50 ≤ P < 0.80)", f"{int((results_df['Severity_Tier'] == 'HIGH').sum())}")
c5.metric("MEDIUM-ALERT", f"{int((results_df['Severity_Tier'] == 'MEDIUM-ALERT').sum())}")

# Visual Diagnostics Row
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Phase-Space Manifold & Flagged Anomalies")
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.5))
    norm = results_df[results_df["Predicted_Label"] == "Normal"]
    abn = results_df[results_df["Predicted_Label"] == "Abnormal"]
    ax1.scatter(norm["Electricity_Consumed"], norm["Avg_Past_Consumption"], c="#2b5c8f", alpha=0.35, s=18, label="Pred: Normal")
    ax1.scatter(abn["Electricity_Consumed"], abn["Avg_Past_Consumption"], c="#d95f02", alpha=0.95, s=45, marker="^", edgecolors="k", label="Pred: Abnormal")
    ax1.set_xlabel("Electricity Consumed (Normalized)")
    ax1.set_ylabel("Avg Past Consumption (9-Period Rolling)")
    ax1.legend(loc="upper left")
    st.pyplot(fig1)

with col_right:
    st.subheader("2. Single-Record XAI Deep-Dive Inspector")
    if len(flagged_df) > 0:
        selected_idx = st.selectbox(
            "Select Flagged Anomaly Row to Inspect:",
            flagged_df.index,
            format_func=lambda idx: f"Row {idx} | {flagged_df.loc[idx, 'Timestamp']} | {flagged_df.loc[idx, 'Severity_Tier']} (P={flagged_df.loc[idx, 'Anomaly_Probability']:.4f})"
        )
        row = results_df.loc[selected_idx]
        st.markdown(f"**Diagnostic Reason:** `{row['Diagnostic_Explanation']}`")
        st.markdown(f"**Data Metrics Snapshot:** `{row['Data_Metrics_Evidence']}`")
        st.markdown(
            f"**Primary Metric Driver:** `{row['Primary_Metric_Driver']}` | "
            f"**NN Attribution Group:** `{row['NN_Top_Attribution_Group']}` ($\Delta P = +{row['NN_Prob_Contribution']:.4f}$)"
        )

        # Compute & plot per-feature Z-score bar chart for the selected row
        feat_df = detector.engineer_features(pd.DataFrame([row]))
        inspect_cols = ["Electricity_Consumed", "Avg_Past_Consumption", "Consumption_Delta", "Mahalanobis_Dist", "Iso_Score"]
        z_vals = [
            (feat_df.iloc[0][c] - detector.normal_stats[c]["mean"]) / (detector.normal_stats[c]["std"] + 1e-8)
            for c in inspect_cols
        ]
        fig2, ax2 = plt.subplots(figsize=(6.5, 2.8))
        bar_colors = ["#d95f02" if abs(z) >= 2.0 else "#2b5c8f" for z in z_vals]
        ax2.barh(inspect_cols, z_vals, color=bar_colors, edgecolor="black")
        ax2.axvline(2.0, color="red", linestyle="--", lw=1, label="±2σ Normal Envelope")
        ax2.axvline(-2.0, color="red", linestyle="--", lw=1)
        ax2.set_xlabel("Z-Score Deviation from Normal Baseline (σ)")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)
    else:
        st.info("No anomalies flagged at the current threshold.")

# Webhook Trigger & Audit Table
st.subheader("3. Flagged Anomaly Audit Table & Webhook Dispatch")
btn_col, dl_col = st.columns([1, 1])
with btn_col:
    if st.button(f"🚀 Dispatch Webhook Alerts for {min_sev}+ Anomalies"):
        dispatcher = AnomalyWebhookDispatcher(webhook_url=webhook_url, min_severity=min_sev)
        sent = dispatcher.dispatch_from_dataframe(flagged_df, dry_run=dry_run_mode)
        st.success(f"Dispatched {len(sent)} {min_sev}+ anomaly alerts (Dry-Run={dry_run_mode}).")

with dl_col:
    st.download_button(
        "⬇️ Download Full Explained Predictions CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="explained_predictions.csv",
        mime="text/csv",
    )

st.dataframe(
    flagged_df[[
        "Timestamp", "Predicted_Label", "Anomaly_Probability", "Severity_Tier",
        "Primary_Metric_Driver", "NN_Top_Attribution_Group", "NN_Prob_Contribution",
        "Data_Metrics_Evidence", "Diagnostic_Explanation"
    ]],
    use_container_width=True
)
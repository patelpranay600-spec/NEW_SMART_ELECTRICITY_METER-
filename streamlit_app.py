# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Smart Meter XAI Live SOC", layout="wide")

st.title("⚡ Smart Meter Anomaly Detection — Live Cloud SOC")
st.markdown("Connected to live **FastAPI Backend** deployed on Render.")

# Sidebar for API URL configuration
st.sidebar.header("🔌 API Configuration")

default_api_url = "https://new-smart-electricity-meter.onrender.com/" 
api_url = st.sidebar.text_input("FastAPI Backend URL", value=default_api_url)

# Health Check
try:
    health_resp = requests.get(f"{api_url}/", timeout=5)
    if health_resp.status_code == 200:
        h_data = health_resp.json()
        st.sidebar.success(f"API Online (Model Loaded: {h_data.get('model_loaded', False)})")
    else:
        st.sidebar.warning("API reachable but returned non-200 status.")
except Exception as e:
    st.sidebar.error(f"Could not connect to API: {e}")

tab1, tab2 = st.tabs(["🔍 Real-Time Single Meter Ingestion", "📂 Batch CSV File Ingestion"])

with tab1:
    st.header("Real-Time Telemetry Scoring & XAI Diagnostics")
    
    col1, col2 = st.columns(2)
    with col1:
        timestamp = st.text_input("Timestamp", value="2024-04-15 04:00:00")
        electricity_consumed = st.number_input("Electricity Consumed", value=0.812, min_value=0.0, max_value=1.5, step=0.01)
        temperature = st.number_input("Temperature", value=0.524, min_value=0.0, max_value=1.5, step=0.01)
        humidity = st.number_input("Humidity", value=0.498, min_value=0.0, max_value=1.5, step=0.01)
    with col2:
        wind_speed = st.number_input("Wind Speed", value=0.465, min_value=0.0, max_value=1.5, step=0.01)
        avg_past_consumption = st.number_input("Avg Past Consumption (Rolling)", value=0.458, min_value=0.0, max_value=1.5, step=0.01)
        operating_threshold = st.slider("Probability Threshold (T)", 0.10, 0.90, 0.35, 0.01)
        webhook_url = st.text_input("Webhook URL (Optional - Slack/Teams)", value="")

    if st.button("🚀 Analyze Telemetry Reading", type="primary"):
        payload = {
            "timestamp": timestamp,
            "electricity_consumed": electricity_consumed,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "avg_past_consumption": avg_past_consumption,
            "operating_threshold": operating_threshold,
            "webhook_url": webhook_url
        }
        try:
            resp = requests.post(f"{api_url}/v1/predict_single", json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                pred = data["prediction"]
                sev = data["severity_tier"]
                
                if pred == "Abnormal":
                    st.error(f"🚨 ANOMALY DETECTED ({sev}) — Probability: {data['anomaly_probability']:.4f}")
                else:
                    st.success(f"✅ NORMAL GRID OPERATION — Probability: {data['anomaly_probability']:.4f}")
                
                st.markdown("### 🧠 XAI Diagnostic Audit")
                st.info(f"**Root-Cause Diagnosis:** {data['diagnostic_explanation']}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Primary Metric Driver", data["primary_metric_driver"])
                m2.metric("Attribution Group", data["nn_attribution_group"])
                m3.metric("Probability Impact (ΔP)", f"+{data['nn_prob_contribution_delta_p']:.4f}")
                
                st.text(f"Telemetry Evidence Snapshot: {data['data_metrics_evidence']}")
            else:
                st.error(f"API Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

with tab2:
    st.header("Batch CSV Telemetry Ingestion")
    uploaded_file = st.file_uploader("Upload Smart Meter CSV", type=["csv"])
    batch_threshold = st.slider("Batch Threshold (T)", 0.10, 0.90, 0.35, 0.01, key="batch_t")
    batch_webhook = st.text_input("Webhook URL for Batch Alerts", value="", key="batch_wh")
    
    if uploaded_file is not None and st.button("🚀 Process Batch CSV"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        params = {"threshold": batch_threshold, "webhook_url": batch_webhook, "min_severity": "HIGH"}
        try:
            resp = requests.post(f"{api_url}/v1/predict_batch_csv", files=files, params=params, timeout=30)
            if resp.status_code == 200:
                b_data = resp.json()
                st.success(f"Processed {b_data['total_records_processed']:,} records. Flagged {b_data['total_anomalies_flagged']:,} anomalies.")
                
                if b_data['flagged_summary']:
                    st.markdown("### Flagged Anomalies Preview")
                    st.dataframe(pd.DataFrame(b_data['flagged_summary']))
                
                dl_url = f"{api_url}{b_data['download_report_url']}"
                st.markdown(f"📥 [Download Full Explained Audit CSV Report]({dl_url})")
            else:
                st.error(f"Batch API Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"Batch request failed: {e}")
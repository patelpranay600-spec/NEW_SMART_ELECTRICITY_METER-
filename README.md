# ⚡ Smart Meter Anomaly Detection — Explainable Neural Network (XAI) & SOC System

A production-grade, end-to-end Deep Neural Network (DNN) anomaly detection and Explainable AI (XAI) Security Operations Center (SOC) platform built for smart grid telemetry. 

The system ingests 30-minute interval smart meter electrical consumption and weather readings, evaluates them against a robust elliptical phase-space manifold, classifies anomalies using a multilayer perceptron ($128 \to 64 \to 32$), calculates exact metric $Z$-scores and counterfactual attributions ($\Delta P$), and automatically dispatches structured webhook alerts to Slack, Microsoft Teams, or PagerDuty.

---

## 🏛️ System Architecture & Tech Stack

- **Deep Learning Core**: Scikit-Learn Multilayer Perceptron (`MLPClassifier`) with L2 regularization ($\alpha = 10^{-3}$) and early stopping.
- **Unsupervised Manifolds**: Empirical Covariance (Mahalanobis Distance $D_M^2$) and Isolation Forest density scoring.
- **Explainable AI (XAI)**: Automated metric $Z$-score tracking ($\sigma$ deviation from normal baseline) and counterfactual neural network probability attributions ($\Delta P$).
- **Backend API**: FastAPI microservice supporting real-time single inference and asynchronous batch CSV ingestion with background webhook dispatchers.
- **Frontend Dashboard**: Streamlit cloud-ready frontend interface connecting to the remote FastAPI backend.
- **Alerting & Webhooks**: Automated JSON payload dispatcher with exponential backoff retries for Slack, Teams, Discord, or SOC SIEM endpoints.
- **Containerization**: Docker & Docker Compose multi-service orchestration.

---

## 📂 Project File & Directory Structure

```text
smart_meter_xai_system/
│
├── data/
│   ├── smart_meter_data.csv             # Raw 5,000-row telemetry dataset
│   ├── train_smart_meter.csv            # Stratified training split (80%)
│   └── test_smart_meter.csv             # Locked holdout test split (20%)
│
├── artifacts/
│   └── explainable_nn_detector.joblib   # Serialized unified feature pipeline & NN model
│
├── src/
│   ├── __init__.py                      # Package initializer
│   ├── engine.py                        # Core feature engineering, NN model, and XAI explainer
│   └── webhooks.py                      # Automated Slack / Teams / SOC webhook dispatcher
│
├── scripts/
│   ├── prepare_splits.py                # Generates train/test CSV splits
│   └── train.py                         # Trains the neural network and saves artifacts
│
├── app_api.py                           # FastAPI backend REST microservice
├── streamlit_app.py                     # Streamlit cloud frontend client UI
├── requirements.txt                     # Pinned Python dependencies
├── Dockerfile                           # Production container definition
└── docker-compose.yml                   # Multi-service orchestration definition
```

---

## 🚀 Quick Start & Local Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/smart-meter-xai-system.git
   cd smart-meter-xai-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Neural Network & Generate Artifacts**:
   ```bash
   python -m scripts.train
   ```

4. **Launch FastAPI Backend Locally**:
   ```bash
   uvicorn app_api:app --host 0.0.0.0 --port 8000 --reload
   ```
   * Access interactive Swagger API Docs at: `http://localhost:8000/docs`

5. **Launch Streamlit Frontend Locally**:
   ```bash
   streamlit run dashboard.py
   ```

---

## 🐳 Docker Production Deployment

To run the complete FastAPI backend stack inside a container:

```bash
# Build and start container in background mode
docker-compose up --build -d
```

---

## 🔌 API Endpoints Reference

### 1. Health Check (`GET /`)
Returns system status and model load confirmation.
* **URL**: `https://your-render-url.onrender.com/`

### 2. Real-Time Single Meter Inference (`POST /v1/predict_single`)
Scores a single telemetry reading and returns predictions, severity tiers, $Z$-score evidence, and XAI root-cause explanations.
* **Payload Example**:
  ```json
  {
    "timestamp": "2024-04-15 04:00:00",
    "electricity_consumed": 0.812,
    "temperature": 0.524,
    "humidity": 0.498,
    "wind_speed": 0.465,
    "avg_past_consumption": 0.458,
    "operating_threshold": 0.35,
    "webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ"
  }
  ```

### 3. Batch CSV Ingestion (`POST /v1/predict_batch_csv`)
Uploads a CSV file of smart meter records, scores all rows, triggers background webhook alerts for flagged anomalies, and generates a downloadable audit report.

---

## 📊 Anomaly Severity Tiers & Root-Cause Categories

- **`CRITICAL` ($P \ge 0.80$)**: Immediate automated webhook dispatch & SOC breaker/fraud flag.
- **`HIGH` ($0.50 \le P < 0.80$)**: Priority technician review queue.
- **`MEDIUM-ALERT` ($0.35 \le P < 0.50$)**: Logged for daily grid analytics.
- **Root-Cause Classifications**:
  1. *Meter Dropout / Zero-Load Outage*
  2. *Acute Consumption Surge / Overload*
  3. *Sustained Rolling High-Load Anomaly*
  4. *Sustained Low-Load / Bypass Tamper*
  5. *Abrupt Load Curtailment*
  6. *Phase-Space Boundary Violation*

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

# app_api.py
import os
import traceback
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from src.webhooks import AnomalyWebhookDispatcher

app = FastAPI(
    title="Smart Meter XAI Neural Network API",
    version="1.0.0",
    description="Production-grade real-time & batch CSV anomaly detection with XAI metrics and webhook dispatching."
)

MODEL_PATH = "artifacts/explainable_nn_detector.joblib"

def get_detector():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

class SingleReading(BaseModel):
    timestamp: str = Field(..., example="2024-04-15 04:00:00")
    electricity_consumed: float = Field(..., ge=0.0, le=1.5, example=0.812)
    temperature: float = Field(..., ge=0.0, le=1.5, example=0.524)
    humidity: float = Field(..., ge=0.0, le=1.5, example=0.498)
    wind_speed: float = Field(..., ge=0.0, le=1.5, example=0.465)
    avg_past_consumption: float = Field(..., ge=0.0, le=1.5, example=0.458)
    operating_threshold: float = Field(0.35, ge=0.1, le=0.9, example=0.35)
    webhook_url: str = Field("", example="https://hooks.slack.com/services/XXX/YYY/ZZZ")

@app.get("/")
def health_check():
    det = get_detector()
    return {
        "status": "online",
        "model_loaded": det is not None,
        "artifact_exists": os.path.exists(MODEL_PATH),
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }

@app.post("/v1/predict_single")
def predict_single(reading: SingleReading, background_tasks: BackgroundTasks):
    try:
        det = get_detector()
        if det is None:
            raise HTTPException(
                status_code=503, 
                detail=f"Model artifact not found at {MODEL_PATH}. Run training script first."
            )
        
        det.threshold = reading.operating_threshold
        
        input_df = pd.DataFrame([{
            "Timestamp": reading.timestamp,
            "Electricity_Consumed": reading.electricity_consumed,
            "Temperature": reading.temperature,
            "Humidity": reading.humidity,
            "Wind_Speed": reading.wind_speed,
            "Avg_Past_Consumption": reading.avg_past_consumption,
            "Anomaly_Label": "Unknown"
        }])
        
        os.makedirs("data", exist_ok=True)
        tmp_path = "data/temp_single_eval.csv"
        input_df.to_csv(tmp_path, index=False)
        
        res_df = det.explain_and_predict_csv(tmp_path, output_csv_path="")
        row = res_df.iloc[0]
        
        result = {
            "timestamp": str(row["Timestamp"]),
            "prediction": str(row["Predicted_Label"]),
            "anomaly_probability": float(row["Anomaly_Probability"]),
            "severity_tier": str(row["Severity_Tier"]),
            "primary_metric_driver": str(row["Primary_Metric_Driver"]),
            "nn_attribution_group": str(row["NN_Top_Attribution_Group"]),
            "nn_prob_contribution_delta_p": float(row["NN_Prob_Contribution"]),
            "data_metrics_evidence": str(row["Data_Metrics_Evidence"]),
            "diagnostic_explanation": str(row["Diagnostic_Explanation"]),
        }
        
        if reading.webhook_url and row["Severity_Tier"] in ["HIGH", "CRITICAL"]:
            dispatcher = AnomalyWebhookDispatcher(webhook_url=reading.webhook_url, min_severity="HIGH")
            background_tasks.add_task(dispatcher.dispatch_from_dataframe, pd.DataFrame([row]), dry_run=False)
            
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        print("ERROR IN PREDICT_SINGLE:\n", error_trace)
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": error_trace})
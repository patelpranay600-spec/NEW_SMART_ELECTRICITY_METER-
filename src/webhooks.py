# src/webhooks.py
import json
import logging
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmartMeterWebhook")

class AnomalyWebhookDispatcher:
    """Dispatches structured XAI anomaly alerts with retry logic to SOC webhooks."""

    def __init__(self, webhook_url: str, min_severity: str = "HIGH", max_retries: int = 3):
        self.webhook_url = webhook_url
        self.severity_rank = {"NORMAL": 0, "MEDIUM-ALERT": 1, "HIGH": 2, "CRITICAL": 3}
        self.min_rank = self.severity_rank.get(min_severity, 2)
        self.max_retries = max_retries

    def build_payload(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "alert_id": f"SM-{row['Timestamp'].replace(' ', 'T').replace(':', '')}",
            "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "meter_timestamp": str(row["Timestamp"]),
            "prediction": str(row["Predicted_Label"]),
            "severity_tier": str(row["Severity_Tier"]),
            "anomaly_probability": float(row["Anomaly_Probability"]),
            "telemetry": {
                "electricity_consumed": round(float(row["Electricity_Consumed"]), 4),
                "avg_past_consumption": round(float(row["Avg_Past_Consumption"]), 4),
                "temperature": round(float(row["Temperature"]), 4),
                "humidity": round(float(row["Humidity"]), 4),
                "wind_speed": round(float(row["Wind_Speed"]), 4),
            },
            "xai_attribution": {
                "primary_metric_driver": str(row["Primary_Metric_Driver"]),
                "nn_top_attribution_group": str(row["NN_Top_Attribution_Group"]),
                "nn_prob_contribution_delta_p": float(row["NN_Prob_Contribution"]),
                "data_metrics_evidence": str(row["Data_Metrics_Evidence"]),
                "diagnostic_explanation": str(row["Diagnostic_Explanation"]),
            },
            # Slack / Teams compatible summary text block
            "text": (
                f"🚨 *[{row['Severity_Tier']}] Smart Meter Anomaly Detected* at `{row['Timestamp']}`\n"
                f"• *Probability:* `{row['Anomaly_Probability']:.4f}` | *Primary Driver:* `{row['Primary_Metric_Driver']}`\n"
                f"• *Metrics:* `{row['Data_Metrics_Evidence']}`\n"
                f"• *Diagnosis:* {row['Diagnostic_Explanation']}"
            ),
        }

    def send_alert(self, payload: Dict[str, Any], dry_run: bool = False) -> bool:
        if dry_run or not self.webhook_url:
            logger.info(f"[DRY-RUN WEBHOOK] {json.dumps(payload, indent=2)}")
            return True

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        logger.info(f"Alert {payload['alert_id']} delivered (HTTP {resp.status}).")
                        return True
            except urllib.error.URLError as e:
                wait = 2 ** (attempt - 1)
                logger.warning(f"Webhook attempt {attempt}/{self.max_retries} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        logger.error(f"Failed to deliver alert {payload['alert_id']} after {self.max_retries} attempts.")
        return False

    def dispatch_from_dataframe(self, explained_df: pd.DataFrame, dry_run: bool = False) -> List[Dict[str, Any]]:
        dispatched = []
        for _, row in explained_df.iterrows():
            rank = self.severity_rank.get(str(row["Severity_Tier"]), 0)
            if rank >= self.min_rank:
                payload = self.build_payload(row)
                if self.send_alert(payload, dry_run=dry_run):
                    dispatched.append(payload)
        return dispatched
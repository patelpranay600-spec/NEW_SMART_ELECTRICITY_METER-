# scripts/train.py
import os
import joblib
from src.engine import ExplainableSmartMeterNN

if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    detector = ExplainableSmartMeterNN(threshold=0.35)
    detector.fit("data/train_smart_meter.csv")

    joblib.dump(detector, "artifacts/explainable_nn_detector.joblib")
    print("Model trained and saved to artifacts/explainable_nn_detector.joblib")

    # Verify on test split
    out_df = detector.explain_and_predict_csv(
        "data/test_smart_meter.csv", "data/explained_predictions.csv"
    )
    print(f"Test evaluation complete: {len(out_df[out_df['Predicted_Label'] == 'Abnormal'])} anomalies flagged.")
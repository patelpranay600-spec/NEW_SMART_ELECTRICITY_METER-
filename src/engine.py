# step2_explainable_nn_engine.py
import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

class ExplainableSmartMeterNN:
    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self.cov_estimator = EmpiricalCovariance()
        self.iso_estimator = IsolationForest(contamination=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )
        self.core_cols = ["Electricity_Consumed", "Avg_Past_Consumption"]
        self.feature_names = [
            "Electricity_Consumed", "Temperature", "Humidity", "Wind_Speed", "Avg_Past_Consumption",
            "Consumption_Delta", "Abs_Consumption_Delta", "Consumption_Ratio",
            "Temp_Humidity_Index", "Weather_Load_Diff",
            "Elec_Sq", "Avg_Past_Sq", "Elec_x_Avg_Past",
            "Mahalanobis_Dist", "Iso_Score"
        ]
        self.normal_stats = {}
        self.normal_scaled_mean = None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["Consumption_Delta"] = d["Electricity_Consumed"] - d["Avg_Past_Consumption"]
        d["Abs_Consumption_Delta"] = d["Consumption_Delta"].abs()
        d["Consumption_Ratio"] = d["Electricity_Consumed"] / (d["Avg_Past_Consumption"] + 1e-4)
        d["Temp_Humidity_Index"] = d["Temperature"] * d["Humidity"]
        d["Weather_Load_Diff"] = d["Electricity_Consumed"] - (
            0.5 * d["Temperature"] + 0.3 * d["Humidity"] + 0.2 * d["Wind_Speed"]
        )
        d["Elec_Sq"] = d["Electricity_Consumed"] ** 2
        d["Avg_Past_Sq"] = d["Avg_Past_Consumption"] ** 2
        d["Elec_x_Avg_Past"] = d["Electricity_Consumed"] * d["Avg_Past_Consumption"]
        d["Mahalanobis_Dist"] = self.cov_estimator.mahalanobis(d[self.core_cols])
        d["Iso_Score"] = -self.iso_estimator.score_samples(d[self.core_cols])
        return d[self.feature_names]

    def fit(self, train_csv_path: str):
        df_train = pd.read_csv(train_csv_path)
        y_train = (df_train["Anomaly_Label"] == "Abnormal").astype(int).values

        # 1. Fit manifold boundary estimators strictly on Normal training data
        normal_df = df_train[y_train == 0]
        self.cov_estimator.fit(normal_df[self.core_cols])
        self.iso_estimator.fit(df_train[self.core_cols])

        # 2. Engineer features and scale
        X_feat = self.engineer_features(df_train)
        X_scaled = self.scaler.fit_transform(X_feat)
        self.normal_scaled_mean = X_scaled[y_train == 0].mean(axis=0)

        # 3. Record Normal Baseline distribution stats for Z-score & percentile explanations
        X_feat_norm = X_feat[y_train == 0]
        for col in self.feature_names:
            self.normal_stats[col] = {
                "mean": float(X_feat_norm[col].mean()),
                "std": float(X_feat_norm[col].std()),
                "p01": float(X_feat_norm[col].quantile(0.01)),
                "p99": float(X_feat_norm[col].quantile(0.99)),
            }

        # 4. Train Deep Neural Network
        self.model.fit(X_scaled, y_train)
        return self

    # step3_ingest_and_explain_csv.py (Continuation of ExplainableSmartMeterNN)
    def explain_and_predict_csv(self, test_csv_path: str, output_csv_path: str = "explained_predictions.csv") -> pd.DataFrame:
        df_test = pd.read_csv(test_csv_path)
        X_feat = self.engineer_features(df_test)
        X_scaled = self.scaler.transform(X_feat)

        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        # 1. Calculate Z-scores against Normal Baseline for all features
        z_scores = pd.DataFrame(index=X_feat.index)
        for col in self.feature_names:
            mu, sd = self.normal_stats[col]["mean"], self.normal_stats[col]["std"] + 1e-8
            z_scores[col] = (X_feat[col] - mu) / sd

        # 2. Counterfactual Neural Network Probability Attribution (ΔP)
        feature_groups = {
            "Instantaneous_Load": ["Electricity_Consumed", "Elec_Sq", "Weather_Load_Diff"],
            "Rolling_Past_Avg": ["Avg_Past_Consumption", "Avg_Past_Sq"],
            "Load_Delta_Ratio": ["Consumption_Delta", "Abs_Consumption_Delta", "Consumption_Ratio", "Elec_x_Avg_Past"],
            "Manifold_Outlier_Score": ["Mahalanobis_Dist", "Iso_Score"],
            "Weather_Context": ["Temperature", "Humidity", "Wind_Speed", "Temp_Humidity_Index"],
        }
        group_impacts = pd.DataFrame(index=X_feat.index)
        for grp_name, cols in feature_groups.items():
            idx_list = [self.feature_names.index(c) for c in cols]
            X_ablated = X_scaled.copy()
            X_ablated[:, idx_list] = self.normal_scaled_mean[idx_list]
            ablated_probs = self.model.predict_proba(X_ablated)[:, 1]
            group_impacts[grp_name] = np.maximum(0.0, probs - ablated_probs)

        # 3. Build Per-Row Diagnostic Explanation & Metric Evidence
        key_metrics = ["Electricity_Consumed", "Avg_Past_Consumption", "Consumption_Delta", "Mahalanobis_Dist", "Iso_Score"]
        reasons, drivers, evidences, severities = [], [], [], []

        for i in range(len(df_test)):
            prob, pred = probs[i], preds[i]
            elec = X_feat.iloc[i]["Electricity_Consumed"]
            avg_p = X_feat.iloc[i]["Avg_Past_Consumption"]
            delta = X_feat.iloc[i]["Consumption_Delta"]
            md = X_feat.iloc[i]["Mahalanobis_Dist"]
            iso = X_feat.iloc[i]["Iso_Score"]

            z_elec = z_scores.iloc[i]["Electricity_Consumed"]
            z_avg = z_scores.iloc[i]["Avg_Past_Consumption"]
            z_delta = z_scores.iloc[i]["Consumption_Delta"]
            z_md = z_scores.iloc[i]["Mahalanobis_Dist"]

            top_feat = z_scores.iloc[i][key_metrics].abs().idxmax()
            top_z = z_scores.iloc[i][top_feat]

            if pred == 1:
                if elec <= 0.05 and avg_p >= 0.12:
                    cat = "Meter Dropout / Zero-Load Outage"
                    exp = f"Near-zero load ({elec:.4f}) despite normal rolling history ({avg_p:.4f}), triggering a sharp drop (Delta={delta:+.4f}, Z={z_delta:+.2f}σ)."
                elif elec >= 0.72 and delta > 0.15:
                    cat = "Acute Consumption Surge / Overload"
                    exp = f"Instantaneous load ({elec:.4f}, Z={z_elec:+.2f}σ) spiked above rolling baseline ({avg_p:.4f}), pushing Mahalanobis distance to {md:.2f} (Normal p99={self.normal_stats['Mahalanobis_Dist']['p99']:.2f})."
                elif avg_p >= 0.78:
                    cat = "Sustained Rolling High-Load Anomaly"
                    exp = f"9-period rolling load ({avg_p:.4f}, Z={z_avg:+.2f}σ) exceeds normal p99 ceiling ({self.normal_stats['Avg_Past_Consumption']['p99']:.4f}), with active load at {elec:.4f}."
                elif avg_p <= 0.14:
                    cat = "Sustained Low-Load / Bypass Tamper"
                    exp = f"Rolling baseline ({avg_p:.4f}, Z={z_avg:+.2f}σ) and active load ({elec:.4f}) sit below the 1st percentile normal floor ({self.normal_stats['Avg_Past_Consumption']['p01']:.4f})."
                elif delta < -0.25:
                    cat = "Abrupt Load Curtailment"
                    exp = f"Active load dropped to {elec:.4f} from a high rolling baseline of {avg_p:.4f} (Delta={delta:+.4f}, Z={z_delta:+.2f}σ, Mahalanobis={md:.2f})."
                else:
                    cat = "Phase-Space Boundary Violation"
                    exp = f"Joint combination of Load ({elec:.4f}) and Rolling Avg ({avg_p:.4f}) violates the elliptical normal manifold (Mahalanobis={md:.2f}, Z={z_md:+.2f}σ; IsoScore={iso:.3f})."
                sev = "CRITICAL" if prob >= 0.80 else ("HIGH" if prob >= 0.50 else "MEDIUM-ALERT")
            else:
                cat = "Normal Grid Operation"
                exp = f"Within normal manifold: Load={elec:.4f} (Z={z_elec:+.2f}σ), Rolling Avg={avg_p:.4f} (Z={z_avg:+.2f}σ), Mahalanobis={md:.2f}."
                sev = "NORMAL"

            snapshot = (
                f"Elec={elec:.3f} ({z_elec:+.1f}σ) | AvgPast={avg_p:.3f} ({z_avg:+.1f}σ) | "
                f"Delta={delta:+.3f} ({z_delta:+.1f}σ) | Mahalanobis={md:.2f} ({z_md:+.1f}σ) | IsoScore={iso:.3f}"
            )
            reasons.append(f"[{cat}] {exp}")
            drivers.append(f"{top_feat} (Z={top_z:+.2f}σ)")
            evidences.append(snapshot)
            severities.append(sev)

        # 4. Append Explainability Columns and Export CSV
        out_df = df_test.copy()
        out_df["Predicted_Label"] = np.where(preds == 1, "Abnormal", "Normal")
        out_df["Anomaly_Probability"] = np.round(probs, 5)
        out_df["Severity_Tier"] = severities
        out_df["Primary_Metric_Driver"] = drivers
        out_df["NN_Top_Attribution_Group"] = group_impacts.idxmax(axis=1)
        out_df["NN_Prob_Contribution"] = np.round(group_impacts.max(axis=1), 4)
        out_df["Data_Metrics_Evidence"] = evidences
        out_df["Diagnostic_Explanation"] = reasons

        if output_csv_path and isinstance(output_csv_path, str) and len(output_csv_path.strip()) > 0:
             out_df.to_csv(output_csv_path, index=False)
        return out_df

if __name__ == "__main__":
    detector = ExplainableSmartMeterNN(threshold=0.35).fit("train_smart_meter.csv")
    joblib.dump(detector, "explainable_nn_detector.joblib")
    results = detector.explain_and_predict_csv("test_smart_meter.csv", "explained_predictions.csv")
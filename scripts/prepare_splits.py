# step1_prepare_csv_splits.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("smart_meter_data.csv")
df["Target"] = (df["Anomaly_Label"] == "Abnormal").astype(int)

# Stratified 80/20 split to create a realistic unseen test CSV
train_df, test_df = train_test_split(
    df, test_size=0.20, stratify=df["Target"], random_state=42
)

train_df.drop(columns=["Target"]).to_csv("train_smart_meter.csv", index=False)
test_df.drop(columns=["Target"]).to_csv("test_smart_meter.csv", index=False)
print(f"Saved train_smart_meter.csv ({len(train_df)}) & test_smart_meter.csv ({len(test_df)})")
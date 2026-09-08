import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
import joblib
import time

def train_extratrees():
    print("Loading dataset...")
    csv_path = "dataset_openai.csv"
    if not os.path.exists(csv_path):
        print(f"Dataset nao encontrado em: {csv_path}")
        print("Execute primeiro: python train_teacher.py && python generate_dataset.py")
        return
    df = pd.read_csv(csv_path)

    # States (17) and Actions (6) — layout canônico obs_*/action_* com fallback posicional
    obs_cols = [c for c in df.columns if c.startswith('obs_') and not c.startswith('next_obs_')]
    act_cols = [c for c in df.columns if c.startswith('action_')]
    if obs_cols and act_cols:
        X = df[obs_cols].values
        y = df[act_cols].values
    else:
        X = df.iloc[:, :17].values
        y = df.iloc[:, 34:40].values
    
    print("Training Extra Trees Regressor...")
    start_time = time.time()
    
    # n_estimators=100 is usually enough for Extra Trees
    # n_jobs=-1 uses all CPU cores
    model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    print("Saving model...")
    joblib.dump(model, "extratrees_model.pkl")
    print("Model saved to extratrees_model.pkl.")

if __name__ == "__main__":
    train_extratrees()

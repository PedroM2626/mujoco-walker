import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import joblib
import time

def train_extratrees():
    print("Loading dataset...")
    df = pd.read_csv("dataset_openai.csv")
    
    # States (17) and Actions (6)
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

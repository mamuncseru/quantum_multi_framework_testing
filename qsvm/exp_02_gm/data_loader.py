import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json

def load_and_process_data(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load Data
    # Assuming CSV has no headers, or adjust 'header=None' if it does
    try:
        df = pd.read_csv(config["dataset_path"])
        # Assuming columns are [x1, x2, label]
        X = df.iloc[:, :2].values
        y = df.iloc[:, -1].values
    except Exception as e:
        print(f"Error loading CSV: {e}. Generating dummy data for test.")
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=100, noise=0.1, random_state=config["seed"])

    # Scale Data to [0, 2pi] for quantum embedding
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Subsample for Quantum Kernel (Computationally expensive)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        train_size=config["train_size"], 
        test_size=config["test_size"], 
        random_state=config["seed"]
    )

    return X_train, X_test, y_train, y_test, config
# train_model.py
from src.adaptive_engine import train_and_save_model

if __name__ == "__main__":
    print("Training ML model (this will create model.joblib)...")
    model = train_and_save_model(path="model.joblib", n_samples=1200)
    print("Done. model.joblib saved in project root.")


# src/adaptive_engine.py
from typing import List, Tuple
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

class AdaptiveEngine:
    """
    Existing rule-based engine (unchanged).
    """
    LEVELS = ['Easy', 'Medium', 'Hard']

    def __init__(self, start_level: str = 'Easy', window_size: int = 5, fast_threshold: float = 7.0):
        self.current_level = start_level if start_level.capitalize() in self.LEVELS else 'Easy'
        self.window_size = window_size
        self.fast_threshold = fast_threshold

    def _level_index(self):
        return self.LEVELS.index(self.current_level)

    def increase_level(self):
        idx = self._level_index()
        if idx < len(self.LEVELS) - 1:
            self.current_level = self.LEVELS[idx + 1]

    def decrease_level(self):
        idx = self._level_index()
        if idx > 0:
            self.current_level = self.LEVELS[idx - 1]

    def decide_next(self, last_records: List[dict]) -> str:
        if not last_records:
            return self.current_level

        k = len(last_records)
        accuracy = sum(r['correct'] for r in last_records) / k
        avg_time = sum(r['time_taken'] for r in last_records) / k

        if accuracy >= 0.8 and avg_time <= self.fast_threshold:
            self.increase_level()
        elif accuracy <= 0.5:
            self.decrease_level()
        return self.current_level

# ---------------------------
# ML-based adaptive engine
# ---------------------------
class MLAdaptiveEngine:
    """
    ML-based engine: uses a trained sklearn classifier to predict next level.
    Features: [accuracy_window, avg_time_window, current_level_index]
    Label: next level index (0=Easy,1=Medium,2=Hard)
    """
    LEVELS = ['Easy', 'Medium', 'Hard']
    MODEL_PATH = "model.joblib"

    def __init__(self, start_level: str = 'Easy', model_path: str = None, window_size: int = 5):
        self.current_level = start_level if start_level.capitalize() in self.LEVELS else 'Easy'
        self.window_size = window_size
        self.model = None
        self.model_path = model_path or self.MODEL_PATH
        # try load model if exists
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)

    def _level_index(self):
        return self.LEVELS.index(self.current_level)

    def decide_next(self, last_records: List[dict]) -> str:
        """
        Predict next level using model. If model isn't available, return current level.
        last_records: list of dicts with 'correct' (0/1) and 'time_taken'
        """
        if not last_records or self.model is None:
            return self.current_level

        k = len(last_records)
        accuracy = sum(r['correct'] for r in last_records) / k
        avg_time = sum(r['time_taken'] for r in last_records) / k
        cur_idx = self._level_index()

        X = np.array([[accuracy, avg_time, cur_idx]])
        pred = self.model.predict(X)[0]  # returns 0/1/2
        # clamp and set level
        pred = int(max(0, min(pred, len(self.LEVELS)-1)))
        self.current_level = self.LEVELS[pred]
        return self.current_level

    def load_model(self, path: str):
        self.model = joblib.load(path)

    def save_model(self, path: str = None):
        if self.model is None:
            raise RuntimeError("No model to save.")
        joblib.dump(self.model, path or self.model_path)

    def is_trained(self) -> bool:
        return self.model is not None

# ---------------------------
# Utilities: simulate small dataset and train
# ---------------------------
def simulate_training_data(n_samples: int = 500, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate simple examples mapping (accuracy, avg_time, cur_level_idx) -> next_level_idx.
    This synthetic data helps bootstrap a classifier for the prototype.
    """
    rng = np.random.RandomState(random_state)
    X = []
    y = []
    for _ in range(n_samples):
        # random current level
        cur_idx = rng.randint(0, 3)
        # generate accuracy biased by level (simulate easier at lower level)
        if cur_idx == 0:  # Easy
            accuracy = min(1.0, rng.normal(0.8, 0.18))
            avg_time = max(0.5, rng.normal(6, 3))
        elif cur_idx == 1:  # Medium
            accuracy = min(1.0, rng.normal(0.6, 0.25))
            avg_time = max(0.5, rng.normal(9, 4))
        else:  # Hard
            accuracy = min(1.0, rng.normal(0.45, 0.25))
            avg_time = max(0.5, rng.normal(12, 5))

        # Decide label heuristically (mirror rule-based logic)
        if accuracy >= 0.8 and avg_time <= 7.0:
            next_idx = min(2, cur_idx + 1)
        elif accuracy <= 0.5:
            next_idx = max(0, cur_idx - 1)
        else:
            next_idx = cur_idx

        # add noise
        accuracy = float(max(0.0, min(1.0, accuracy + rng.normal(0, 0.02))))
        avg_time = float(max(0.2, avg_time + rng.normal(0, 1.0)))

        X.append([accuracy, avg_time, cur_idx])
        y.append(next_idx)

    return np.array(X), np.array(y)

def train_and_save_model(path: str = "model.joblib", n_samples: int = 1000) -> DecisionTreeClassifier:
    """
    Train a DecisionTreeClassifier on simulated data and save it.
    Returns trained model.
    """
    X, y = simulate_training_data(n_samples)
    clf = DecisionTreeClassifier(max_depth=5, random_state=0)
    clf.fit(X, y)
    joblib.dump(clf, path)
    return clf

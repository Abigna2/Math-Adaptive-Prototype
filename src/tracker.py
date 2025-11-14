# src/tracker.py
import pandas as pd
from typing import List, Dict

class Tracker:
    """
    Tracks per-question records and computes summary statistics.
    """

    def __init__(self):
        self.records: List[Dict] = []

    def record(self, question: str, level: str, correct: bool, time_taken: float, timestamp: float):
        rec = {
            'question': question,
            'level': level,
            'correct': int(correct),
            'time_taken': float(round(time_taken, 3)),
            'timestamp': timestamp
        }
        self.records.append(rec)

    def last_k(self, k: int = 5):
        return self.records[-k:]

    def accuracy(self, last_k: int = None) -> float:
        records = self.records if last_k is None else self.last_k(last_k)
        if not records:
            return 0.0
        return sum(r['correct'] for r in records) / len(records)

    def avg_time(self, last_k: int = None) -> float:
        records = self.records if last_k is None else self.last_k(last_k)
        if not records:
            return 0.0
        return sum(r['time_taken'] for r in records) / len(records)

    def total_summary(self) -> Dict:
        total = len(self.records)
        if total == 0:
            return {'total': 0, 'correct': 0, 'accuracy': 0.0, 'avg_time': 0.0}
        correct = sum(r['correct'] for r in self.records)
        avg_time = self.avg_time()
        return {
            'total': total,
            'correct': correct,
            'accuracy': round((correct / total) * 100, 2),
            'avg_time': round(avg_time, 3)
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def clear(self):
        self.records = []

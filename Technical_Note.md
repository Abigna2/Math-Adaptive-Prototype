# Technical Note — Adaptive Math Learning Prototype

## Introduction
This prototype demonstrates adaptive difficulty for basic arithmetic practice for children ages 5–10, combining a simple rule-based policy with an optional ML Decision Tree model to choose the next difficulty.

## System Architecture
Four modules:
- `puzzle_generator.py` — generates problems by level (Easy/Medium/Hard).
- `tracker.py` — logs correctness, response time, and timestamps. Provides `to_dataframe()` and summary functions.
- `adaptive_engine.py` — `AdaptiveEngine` (rule-based) and `MLAdaptiveEngine` (Decision Tree wrapper).
- `main.py` — Streamlit UI to orchestrate sessions, training/loading model, and show summary.

## Flow
User → Puzzle Generator → User answers (timer) → Tracker logs → Adaptive Engine (rule / ML) → next problem → summary.

## Adaptive Logic
- **Rule-based**: uses recent K answers (window) and a fast threshold (seconds) to decide:
  - If accuracy high and times fast → increase level
  - If accuracy low or times slow → decrease level
  - Else → maintain
- **ML (Decision Tree)**: trained on simulated samples (features: correctness, response_time, current_level_index). Predicts next level index. Saved/loaded as `model.joblib`.

## Key metrics & influence
- **Accuracy**: primary measure — high → raise difficulty, low → lower.
- **Response time**: measures fluency — very fast correct answers → raise; slow correct → hold or lower.
- **Difficulty history / streaks**: smooth transitions; prevent oscillation.

## Why this approach
- Rule-based is interpretable and robust for limited data.
- ML demonstrates scalable pattern capture and can be retrained with real user data later.
- Lightweight, easy to explain, and practical for a 1–2 page deliverable.

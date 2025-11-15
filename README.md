🧠 Overview

This project is an Adaptive Math Learning System built using Python, Streamlit, and a simple adaptive difficulty algorithm, with optional machine learning–based difficulty prediction.

It generates arithmetic problems for learners and dynamically adjusts the difficulty based on performance, making learning personalized and engaging.

🎥 Demo Video

Watch the full demonstration here:

🔗 [https://drive.google.com/file/d/your-drive-link-here/view](https://drive.google.com/file/d/1um-ITHQ5PfJHLiCSucpTMSa1k2Fr69CS/view?usp=sharing)

🚀 Features

🎯 Adaptive difficulty (Easy → Medium → Hard)

⏱️ Performance tracking (accuracy, time taken, per-question history)

🤖 Optional ML mode using a Decision Tree

📊 Session summary with charts and downloadable CSV

🔢 Multiple question types and level progression

🧩 Child-friendly UI built in Streamlit

🛠️ Tech Stack

Python 3.x

Streamlit (UI)

scikit-learn (ML model)

pandas / numpy

joblib

📂 Project Structure
math-adaptive-prototype/submission_real.zip
│
├── src/
│   ├── main.py
│   ├── adaptive_engine.py
│   ├── puzzle_generator.py
│   ├── tracker.py
│   └── train_model.py
│
├── architecture_diagram.pdf
├── Technical_Note.md
├── requirements.txt
├── README.md


⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Abigna2/math-adaptive-prototype.git
cd math-adaptive-prototype

2️⃣ Create a virtual environment
python -m venv venv

3️⃣ Activate the environment

Windows:

.\venv\Scripts\activate

4️⃣ Install dependencies
pip install -r requirements.txt

5️⃣ Run the application
python -m streamlit run src/main.py

🧠 How Adaptation Works
The system adjusts difficulty using:
✔ Rule-based logic

Correct + fast → Increase level
Incorrect → Decrease level
Balanced difficulty curve

✔ Optional ML mode
Decision Tree predicts the next difficulty
Model trained on simulated learner performance data
Can be toggled in UI

📊 Session Summary

After the session ends, the app shows:
Accuracy
Total attempts
Average time
Difficulty progression
Downloadable CSV
Performance charts

🧩 Architecture Diagram
Included in the repository:
architecture_diagram.pdf

Architecture

Adaptive logic explanation
    |
ML approach
    |
Metrics
    |
Motivation

🤝 Contribution

Contributions are welcome. Fork → modify → pull request!

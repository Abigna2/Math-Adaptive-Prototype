# src/main.py
import streamlit as st
import time
import pandas as pd

from puzzle_generator import PuzzleGenerator
from tracker import Tracker
from adaptive_engine import AdaptiveEngine, MLAdaptiveEngine, train_and_save_model, simulate_training_data

st.set_page_config(page_title="Math Adventures — Adaptive", layout="centered")

# --- Session state init ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.name = ""
    st.session_state.generator = PuzzleGenerator()
    st.session_state.tracker = Tracker()
    st.session_state.engine = AdaptiveEngine(start_level='Easy', window_size=5, fast_threshold=7.0)
    st.session_state.ml_engine = None
    st.session_state.mode = "Rule-based"
    st.session_state.current_question = None
    st.session_state.current_answer = None
    st.session_state.awaiting_answer = False
    st.session_state.start_monotonic = None
    st.session_state.start_wallclock = None
    st.session_state.feedback = ""
    st.session_state.show_summary = False
    st.session_state.num_questions = 10
    st.session_state.asked = 0
    st.session_state._last_submit_key = 0
    st.session_state.model_info = None
    st.session_state.submitted_last = False  # whether last question was submitted
    # NEW: initial difficulty choice (default 'Easy')
    st.session_state.initial_difficulty = 'Easy'

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Session Controls")
    # name and session settings in sidebar
    st.session_state.name = st.text_input("Learner name", value=st.session_state.name)
    st.session_state.num_questions = int(st.number_input("Number of questions", min_value=1, max_value=50, value=st.session_state.num_questions))
    st.session_state.engine.window_size = int(st.number_input("Adaptive window size (K)", min_value=1, max_value=10, value=st.session_state.engine.window_size))
    st.session_state.engine.fast_threshold = float(st.number_input("Fast threshold (s)", min_value=1.0, max_value=30.0, value=st.session_state.engine.fast_threshold))

    st.markdown("---")
    st.header("Initial difficulty")  # <-- Added section
    # 1. Choose the initial difficulty button (radio)
    st.session_state.initial_difficulty = st.radio("Choose starting level", options=["Easy", "Medium", "Hard"], index=["Easy","Medium","Hard"].index(st.session_state.initial_difficulty))

    st.markdown("---")
    st.header("Adaptive mode")
    st.session_state.mode = st.radio("Mode", options=["Rule-based", "ML (Decision Tree)"], index=0 if st.session_state.mode == "Rule-based" else 1)

    st.markdown("---")
    st.header("ML model (optional)")
    st.caption("Train on simulated data to experiment. Decision Tree model.")
    if st.button("Train ML model (simulate + train)"):
        with st.spinner("Training ML model..."):
            try:
                model = train_and_save_model(path="model.joblib", n_samples=1200)
                X_val, y_val = simulate_training_data(n_samples=400, random_state=999)
                acc = model.score(X_val, y_val)
                st.success(f"ML model trained and saved (val acc {acc:.3f})")
                # use engine's current level as start level for ml_engine; engine will be reinitialized on Start
                st.session_state.ml_engine = MLAdaptiveEngine(start_level=st.session_state.engine.current_level, model_path="model.joblib", window_size=st.session_state.engine.window_size)
                st.session_state.model_info = {"accuracy": float(acc)}
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.button("Load ML model from disk"):
        try:
            st.session_state.ml_engine = MLAdaptiveEngine(start_level=st.session_state.engine.current_level, model_path="model.joblib", window_size=st.session_state.engine.window_size)
            if st.session_state.ml_engine.is_trained():
                st.success("Loaded ML model from model.joblib")
                st.session_state.model_info = {"accuracy": None}
            else:
                st.warning("model.joblib exists but could not be loaded as trained model.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    st.markdown("---")
    # Restart only in sidebar
    if st.button("Restart session (keep name)"):
        # Use the selected initial difficulty when re-initializing
        st.session_state.engine = AdaptiveEngine(start_level=st.session_state.initial_difficulty, window_size=st.session_state.engine.window_size, fast_threshold=st.session_state.engine.fast_threshold)
        if st.session_state.ml_engine is not None:
            try:
                st.session_state.ml_engine = MLAdaptiveEngine(start_level=st.session_state.engine.current_level, model_path=st.session_state.ml_engine.model_path, window_size=st.session_state.engine.window_size)
            except Exception:
                pass
        st.session_state.tracker = Tracker()
        st.session_state.current_question = None
        st.session_state.current_answer = None
        st.session_state.awaiting_answer = False
        st.session_state.start_monotonic = None
        st.session_state.start_wallclock = None
        st.session_state.feedback = ""
        st.session_state.show_summary = False
        st.session_state.asked = 0
        st.session_state.submitted_last = False
        st.session_state._last_submit_key += 1

# ---------------- Main area ----------------
st.title("🧠 Math Adventures — Adaptive Practice")

# Start Session button (main area)
start_now = st.button("Start Session")

if start_now:
    # Ensure name is set
    st.session_state.name = st.session_state.name.strip() if st.session_state.name.strip() else "Learner"
    # Re-init engines/tracker and set starting level from the chosen initial difficulty
    st.session_state.engine = AdaptiveEngine(start_level=st.session_state.initial_difficulty, window_size=st.session_state.engine.window_size, fast_threshold=st.session_state.engine.fast_threshold)
    if st.session_state.ml_engine is not None:
        try:
            st.session_state.ml_engine = MLAdaptiveEngine(start_level=st.session_state.engine.current_level, model_path=st.session_state.ml_engine.model_path, window_size=st.session_state.engine.window_size)
        except Exception:
            pass
    st.session_state.tracker = Tracker()
    st.session_state.current_question = None
    st.session_state.current_answer = None
    st.session_state.awaiting_answer = False
    st.session_state.start_monotonic = None
    st.session_state.start_wallclock = None
    st.session_state.feedback = ""
    st.session_state.show_summary = False
    st.session_state.num_questions = int(st.session_state.num_questions)
    st.session_state.asked = 0
    st.session_state._last_submit_key += 1
    st.session_state.submitted_last = False

# ---------- Helpers ----------
def generate_question():
    """Generate a new question and set monotonic start time only once."""
    if st.session_state.current_question is None and st.session_state.asked < st.session_state.num_questions and not st.session_state.show_summary:
        q, ans = st.session_state.generator.generate(st.session_state.engine.current_level)
        st.session_state.current_question = q
        st.session_state.current_answer = ans
        if st.session_state.start_monotonic is None:
            st.session_state.start_monotonic = time.monotonic()
            st.session_state.start_wallclock = time.time()
        st.session_state.awaiting_answer = True
        st.session_state.feedback = ""
        st.session_state.submitted_last = False

# If Start was pressed, generate the first question immediately
if start_now and st.session_state.current_question is None and not st.session_state.show_summary:
    generate_question()

# Show summary if finished
if st.session_state.show_summary or st.session_state.asked >= st.session_state.num_questions:
    st.session_state.show_summary = True
    st.markdown("## 📊 Session Summary")
    summ = st.session_state.tracker.total_summary()
    st.write(f"**Learner:** {st.session_state.name}")
    st.write(f"**Total attempted:** {summ['total']}")
    st.write(f"**Correct:** {summ['correct']} ({summ['accuracy']}%)")
    st.write(f"**Average time/question:** {summ['avg_time']} s")
    st.write(f"**Final recommended level:** {st.session_state.engine.current_level}")

    df = st.session_state.tracker.to_dataframe()
    if not df.empty:
        df_sorted = df.sort_values(by='timestamp').reset_index(drop=True)
        st.markdown("### Performance over time")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(df_sorted['correct'])
            st.caption("1 = correct, 0 = incorrect")
        with c2:
            st.line_chart(df_sorted['time_taken'])
            st.caption("time in seconds")
        st.markdown("### Attempts")
        st.dataframe(df_sorted[['question','level','correct','time_taken','timestamp']].reset_index(drop=True))
        csv = df_sorted.to_csv(index=False)
        st.download_button("Download session CSV", data=csv, file_name="math_adventures_session.csv", mime="text/csv")
    else:
        st.write("No attempts recorded.")
    st.stop()

# ---------- Question UI (form + monotonic timing) ----------
if st.session_state.current_question is not None:
    # Display learner name prominently
    st.subheader(f"👋 {st.session_state.name} — Current level: {st.session_state.engine.current_level}")

    st.markdown("### 🔢 Solve this:")
    st.markdown(f"**{st.session_state.current_question} = ?**")

    # show live elapsed time (monotonic)
    if st.session_state.start_monotonic is not None:
        elapsed_live = time.monotonic() - st.session_state.start_monotonic
        st.caption(f"⏱ Time elapsed: {elapsed_live:.2f} s")

    # Use a form so typing doesn't cause reruns
    form_key = f"answer_form_{st.session_state._last_submit_key}"
    with st.form(key=form_key):
        ans_input = st.text_input("Type your answer here")
        submitted = st.form_submit_button("Submit answer")

    if submitted:
        end_monotonic = time.monotonic()
        end_wallclock = time.time()
        t_taken = end_monotonic - (st.session_state.start_monotonic or end_monotonic)

        # parse user's answer
        try:
            user_ans = float(ans_input)
        except Exception:
            user_ans = None

        correct = False
        if user_ans is not None:
            correct = abs(user_ans - st.session_state.current_answer) < 1e-6

        # record using wall-clock timestamp, but elapsed measured by monotonic
        st.session_state.tracker.record(
            question=st.session_state.current_question,
            level=st.session_state.engine.current_level,
            correct=correct,
            time_taken=round(t_taken, 3),
            timestamp=end_wallclock
        )
        st.session_state.asked += 1

        # --- Difficulty progression: immediate stepwise behavior ---
        # If last answer correct -> increase level; if incorrect -> decrease level.
        if correct:
            st.session_state.engine.increase_level()
        else:
            st.session_state.engine.decrease_level()
        new_level = st.session_state.engine.current_level

        # feedback
        if correct:
            st.session_state.feedback = f"✅ Correct! That took {t_taken:.2f}s. Next level: {new_level}"
        else:
            correct_ans_display = int(st.session_state.current_answer) if float(st.session_state.current_answer).is_integer() else st.session_state.current_answer
            st.session_state.feedback = f"❌ Not quite. Correct answer was **{correct_ans_display}**. That took {t_taken:.2f}s. Next level: {new_level}"

        # cleanup for next question
        st.session_state.current_question = None
        st.session_state.current_answer = None
        st.session_state.awaiting_answer = False
        st.session_state.start_monotonic = None
        st.session_state.start_wallclock = None
        st.session_state._last_submit_key += 1
        st.session_state.submitted_last = True

        # If we've reached the configured total, show summary immediately (Streamlit will rerun and hit summary block)
        if st.session_state.asked >= st.session_state.num_questions:
            st.session_state.show_summary = True

# ---------- Feedback area + Next / End controls ----------
if st.session_state.submitted_last:
    if st.session_state.feedback:
        if "Correct" in st.session_state.feedback:
            st.success(st.session_state.feedback)
        else:
            st.error(st.session_state.feedback)

    cols = st.columns([1,1])
    with cols[0]:
        if st.button("Next Question"):
            # Only generate next question if quota not reached
            if st.session_state.asked >= st.session_state.num_questions:
                st.session_state.show_summary = True
            else:
                # generate next question and start monotonic timer immediately
                generate_question()

    with cols[1]:
        if st.button("End session and show summary"):
            st.session_state.show_summary = True
            st.session_state.submitted_last = False

# If nothing to show, prompt
if st.session_state.current_question is None and not st.session_state.submitted_last and not st.session_state.show_summary:
    st.info("Press **Start Session** to begin. After submitting an answer, press **Next Question** to continue or End to see the summary.")

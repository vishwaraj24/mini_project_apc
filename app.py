"""
StudyMate AI - smart study planner.

Run:  pip install streamlit pandas scikit-learn matplotlib
      streamlit run app.py

AI approach (state this clearly in your report):
  * Default: RULE-BASED weighted priority score (days left, difficulty,
    syllabus remaining, low confidence).
  * Optional: a scikit-learn DecisionTree trained on SYNTHETIC data labelled by
    the same rules - a demo of ML-based priority levels, not real student data.
  * No language model / paid API is used.
"""
import sqlite3
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.tree import DecisionTreeClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

DB = "studymate.db"
W = {"urgency": 0.40, "difficulty": 0.20, "remaining": 0.25, "unconfidence": 0.15}
LEVELS = ["Low", "Medium", "High"]


# ---------------------------------------------------------------- database
def run(sql, params=()):
    with sqlite3.connect(DB) as c:
        c.execute(sql, params)


def query(sql, params=()):
    with sqlite3.connect(DB) as c:
        return pd.read_sql_query(sql, c, params=params)


def init_db():
    run("""CREATE TABLE IF NOT EXISTS subjects(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, exam_date TEXT,
        difficulty INTEGER, confidence INTEGER, manual_pct REAL)""")
    run("""CREATE TABLE IF NOT EXISTS topics(
        id INTEGER PRIMARY KEY AUTOINCREMENT, subject_id INTEGER,
        name TEXT, done INTEGER DEFAULT 0)""")
    run("""CREATE TABLE IF NOT EXISTS study_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT,
        subject_id INTEGER, hours REAL)""")


# ---------------------------------------------------------------- priority
def rule_score(days, difficulty, completion, confidence):
    urgency = np.clip(1 - days / 30, 0, 1)
    return (W["urgency"] * urgency
            + W["difficulty"] * (difficulty - 1) / 4
            + W["remaining"] * (1 - completion / 100)
            + W["unconfidence"] * (5 - confidence) / 4)


def level_from_score(s):
    return "High" if s >= 0.55 else "Medium" if s >= 0.35 else "Low"


@st.cache_resource
def train_model():
    """Decision tree on synthetic study data (labels come from the rule score + noise)."""
    rng = np.random.default_rng(42)
    n = 800
    X = np.column_stack([rng.integers(0, 31, n), rng.integers(1, 6, n),
                         rng.uniform(0, 100, n), rng.integers(1, 6, n)])
    s = rule_score(X[:, 0], X[:, 1], X[:, 2], X[:, 3]) + rng.normal(0, 0.04, n)
    y = [level_from_score(v) for v in s]
    return DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)


def load_subjects(use_ml=False):
    df = query("""SELECT s.*, COUNT(t.id) AS n_topics, COALESCE(SUM(t.done),0) AS n_done
                  FROM subjects s LEFT JOIN topics t ON t.subject_id = s.id
                  GROUP BY s.id""")
    if df.empty:
        return df
    df["days_left"] = (pd.to_datetime(df["exam_date"]).dt.date
                       .apply(lambda d: (d - date.today()).days))
    df["completion"] = np.where(df["n_topics"] > 0,
                                df["n_done"] / df["n_topics"].replace(0, 1) * 100,
                                df["manual_pct"])
    df["score"] = [rule_score(r.days_left, r.difficulty, r.completion, r.confidence)
                   for r in df.itertuples()]
    df["priority"] = df["score"].apply(level_from_score)
    if use_ml and HAS_SKLEARN:
        feats = df[["days_left", "difficulty", "completion", "confidence"]].values
        df["priority"] = train_model().predict(feats)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------- timetable
def build_timetable(df, hours, start, break_min=15):
    active = df[df["days_left"] >= 0]
    if active.empty:
        return pd.DataFrame()
    n = len(active)
    total = max(hours * 60 - break_min * (n - 1), 20 * n)
    weights = active["score"] / active["score"].sum()
    mins = (weights * total / 5).round() * 5
    mins = mins.clip(lower=20)
    if mins.sum() > total:
        mins = ((mins * total / mins.sum()) / 5).round() * 5

    rows, t = [], datetime.combine(date.today(), start)
    for (_, r), m in zip(active.iterrows(), mins):
        end = t + timedelta(minutes=int(m))
        pending = query("SELECT name FROM topics WHERE subject_id=? AND done=0 LIMIT 2",
                        (int(r["id"]),))["name"].tolist()
        if r["days_left"] <= 3:
            task = "Final revision: solve practice questions"
        elif pending:
            task = "Cover: " + ", ".join(pending)
        else:
            task = "Revise key concepts and practise"
        rows.append({"Time": f"{t:%I:%M %p} - {end:%I:%M %p}", "Subject": r["name"],
                     "Task": task, "Priority": r["priority"], "Minutes": int(m)})
        t = end + timedelta(minutes=break_min)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- UI
st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")
init_db()

st.title("📚 StudyMate AI")
st.caption("Smart exam planner: enter your subjects and get a priority-based daily timetable.")

with st.sidebar:
    st.header("Daily settings")
    hours = st.slider("Available study hours today", 1.0, 10.0, 3.0, 0.5)
    start = st.time_input("Start time", datetime.strptime("19:00", "%H:%M").time())
    use_ml = st.toggle("Use ML priority model", value=False, disabled=not HAS_SKLEARN,
                       help="Decision tree trained on synthetic data. Off = rule-based score.")
    st.caption("Mode: " + ("Machine learning (scikit-learn)" if use_ml else "Rule-based weighted score"))

df = load_subjects(use_ml)
tab_setup, tab_plan, tab_prog, tab_stats = st.tabs(
    ["1. Subjects", "2. Timetable", "3. Progress", "4. Analytics"])

# ---- Subjects
with tab_setup:
    with st.form("add_subject", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Subject")
        exam = c2.date_input("Exam date", date.today() + timedelta(days=7), min_value=date.today())
        c3, c4, c5 = st.columns(3)
        diff = c3.slider("Difficulty (1 easy - 5 hard)", 1, 5, 3)
        conf = c4.slider("Your confidence (1 low - 5 high)", 1, 5, 3)
        pct = c5.slider("Syllabus completed (%)", 0, 100, 0,
                        help="Used until you add topics; then topic ticks take over.")
        topics_txt = st.text_area("Topics (one per line, optional)")
        if st.form_submit_button("Add subject") and name.strip():
            run("INSERT INTO subjects(name, exam_date, difficulty, confidence, manual_pct) "
                "VALUES(?,?,?,?,?)", (name.strip(), exam.isoformat(), diff, conf, pct))
            sid = int(query("SELECT MAX(id) AS i FROM subjects")["i"][0])
            for t in [x.strip() for x in topics_txt.splitlines() if x.strip()]:
                run("INSERT INTO topics(subject_id, name) VALUES(?,?)", (sid, t))
            st.rerun()

    if not df.empty:
        show = df[["name", "exam_date", "days_left", "difficulty", "confidence",
                   "completion", "priority"]].copy()
        show["completion"] = show["completion"].round(0).astype(int).astype(str) + "%"
        st.dataframe(show, use_container_width=True, hide_index=True)
        to_del = st.selectbox("Remove a subject", ["-"] + df["name"].tolist())
        if to_del != "-" and st.button("Delete"):
            sid = int(df.loc[df["name"] == to_del, "id"].iloc[0])
            for tbl, col in [("topics", "subject_id"), ("study_log", "subject_id"), ("subjects", "id")]:
                run(f"DELETE FROM {tbl} WHERE {col}=?", (sid,))
            st.rerun()
    else:
        st.info("Add your first subject above.")

# ---- Timetable + revision reminders
with tab_plan:
    if df.empty:
        st.info("Add subjects first.")
    else:
        tt = build_timetable(df, hours, start)
        if tt.empty:
            st.warning("All exams are in the past. Add upcoming exams.")
        else:
            st.subheader(f"Today's timetable ({date.today():%d %b %Y})")
            st.dataframe(tt, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", tt.to_csv(index=False), "timetable.csv")

        st.subheader("🔔 Revision reminders")
        any_rem = False
        for r in df.itertuples():
            if 0 <= r.days_left <= 3:
                st.error(f"**{r.name}**: exam in {r.days_left} day(s). Do a full revision pass today.")
                any_rem = True
            elif r.confidence <= 2 and r.days_left >= 0:
                st.warning(f"**{r.name}**: low confidence. Schedule extra practice problems.")
                any_rem = True
            elif r.days_left >= 0 and r.completion >= 80:
                st.success(f"**{r.name}**: syllabus nearly done. Start revision sessions.")
                any_rem = True
        if not any_rem:
            st.write("Nothing urgent. Keep following the timetable.")

# ---- Progress tracker
with tab_prog:
    if df.empty:
        st.info("Add subjects first.")
    for r in df.itertuples():
        st.markdown(f"**{r.name}** - {r.completion:.0f}% complete")
        st.progress(min(r.completion / 100, 1.0))
        tp = query("SELECT * FROM topics WHERE subject_id=?", (int(r.id),))
        if tp.empty:
            st.caption("No topics added; using the percentage you entered.")
        for t in tp.itertuples():
            new = st.checkbox(t.name, value=bool(t.done), key=f"topic_{t.id}")
            if new != bool(t.done):
                run("UPDATE topics SET done=? WHERE id=?", (int(new), int(t.id)))
                st.rerun()
        with st.expander(f"Add topic to {r.name}"):
            nt = st.text_input("Topic name", key=f"nt_{r.id}")
            if st.button("Add", key=f"nb_{r.id}") and nt.strip():
                run("INSERT INTO topics(subject_id, name) VALUES(?,?)", (int(r.id), nt.strip()))
                st.rerun()

# ---- Analytics
with tab_stats:
    if df.empty:
        st.info("Add subjects first.")
    else:
        with st.form("log_hours", clear_on_submit=True):
            c1, c2 = st.columns(2)
            subj = c1.selectbox("Subject studied", df["name"])
            hrs = c2.number_input("Hours", 0.25, 12.0, 1.0, 0.25)
            if st.form_submit_button("Log study time"):
                sid = int(df.loc[df["name"] == subj, "id"].iloc[0])
                run("INSERT INTO study_log(day, subject_id, hours) VALUES(?,?,?)",
                    (date.today().isoformat(), sid, hrs))
                st.rerun()

        log = query("SELECT day, subject_id, hours FROM study_log")
        today_h = log.loc[log["day"] == date.today().isoformat(), "hours"].sum() if not log.empty else 0
        pending = int((df["n_topics"] - df["n_done"]).sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total hours studied", f"{log['hours'].sum() if not log.empty else 0:.1f}")
        m2.metric("Today vs goal", f"{today_h:.1f} / {hours:.1f} h")
        m3.metric("Pending topics", pending)
        m4.metric("Avg. syllabus done", f"{df['completion'].mean():.0f}%")

        col_a, col_b = st.columns(2)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(df["name"], df["completion"], color="#4C78A8")
        ax.set_xlim(0, 100)
        ax.set_xlabel("% syllabus completed")
        ax.set_title("Subject progress")
        col_a.pyplot(fig)

        if not log.empty:
            daily = log.groupby("day")["hours"].sum()
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            ax2.bar(daily.index, daily.values, color="#59A14F")
            ax2.axhline(hours, color="red", ls="--", label="Daily goal")
            ax2.set_ylabel("Hours")
            ax2.set_title("Study hours per day")
            ax2.legend()
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
            col_b.pyplot(fig2)
        else:
            col_b.info("Log study time to see the daily hours chart.")

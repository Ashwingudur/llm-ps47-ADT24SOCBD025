import os, json, io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY) if API_KEY else None
MODEL = "openai/gpt-oss-20b"

SAMPLE = """date,entry
2026-11-02,Started the week with a long gym session. Felt energetic and got my project plan done.
2026-11-03,Lecture was boring and I had too many pending assignments. Felt a bit stressed.
2026-11-04,Finally fixed the bug in my RAG pipeline! Super happy about it.
2026-11-05,Slept late and felt tired all day. Skipped the evening study session.
2026-11-06,Had a good call with a senior about placements. Feeling motivated.
2026-11-07,Spent time with family. Calm and relaxed day.
2026-11-08,Worried about the upcoming exams and portfolio work piling up.
2026-11-09,Mock interview did not go well. Felt low and doubted myself.
2026-11-10,Talked to a friend and felt better. Did a small coding task for confidence.
2026-11-11,Submitted two assignments on time. Relieved and proud.
2026-11-12,Rainy day, stayed in and read about LLM agents. Peaceful.
2026-11-13,Team meeting went chaotic, deadlines unclear. Annoyed and anxious.
2026-11-14,Went for a run, cleared my head and planned the next week.
"""

MOOD_PROMPT = """You are a journal mood tagger.
Read the journal entry and reply with ONLY a JSON object:
{"mood": "<one word like happy, calm, stressed, sad, anxious, excited, tired, angry>",
 "score": <integer 1-5, 1 = very negative, 3 = neutral, 5 = very positive>}
No extra text."""

SUMMARY_PROMPT = """You are a friendly journaling coach.
Below are one week of journal entries with dates and detected moods.
Write a short summary (max 5 lines) with:
1) overall mood trend of the week
2) 2 highlights (good moments)
3) 1 thing that kept coming up as a stress/worry
Use simple, warm language. Do not invent events."""


def load_journal(text):
    """Split each line at the FIRST comma only, so commas inside entries are safe."""
    rows = []
    for line in text.replace("\ufeff", "").strip().splitlines()[1:]:
        if "," not in line:
            continue
        d, e = line.split(",", 1)
        rows.append((d.strip().strip('"'), e.strip().strip('"')))
    df = pd.DataFrame(rows, columns=["date", "entry"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def ask(system, user, temperature=0.3):
    r = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return r.choices[0].message.content.strip()


def tag_mood(entry):
    raw = ask(MOOD_PROMPT, entry, temperature=0)
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        d = json.loads(raw)
        return str(d["mood"]).lower(), int(d["score"])
    except Exception:
        return "unknown", 3


st.set_page_config(page_title="Daily Journal Summarizer", page_icon="📓")
st.title("📓 Daily Journal Summarizer")
st.caption("Upload a CSV with columns: date, entry  (or use the built-in sample)")

if client is None:
    st.error("GROQ_API_KEY not found. Create a .env file with GROQ_API_KEY=your_key and restart.")
    st.stop()

up = st.file_uploader("Journal CSV", type="csv")
if up is not None:
    text = up.getvalue().decode("utf-8", errors="ignore")
else:
    text = SAMPLE
    st.info("Using built-in sample journal.")

try:
    df = load_journal(text)
except Exception as ex:
    st.error(f"Could not read the CSV. Make sure it has a header 'date,entry' and dates like 2026-11-02. ({ex})")
    st.stop()

st.subheader("Entries")
st.dataframe(df)

if st.button("Analyse"):
    moods, scores = [], []
    bar = st.progress(0, text="Tagging moods...")
    for i, e in enumerate(df["entry"]):
        m, s = tag_mood(str(e))
        moods.append(m)
        scores.append(s)
        bar.progress((i + 1) / len(df))
    out = df.copy()
    out["mood"], out["score"] = moods, scores
    out["week"] = out["date"].dt.to_period("W").astype(str)

    summaries = {}
    for week, g in out.groupby("week"):
        t = "\n".join(f"{r.date.date()} [{r.mood}]: {r.entry}" for r in g.itertuples())
        summaries[week] = (g["score"].mean(), ask(SUMMARY_PROMPT, t, temperature=0.5))

    st.session_state["out"] = out
    st.session_state["summaries"] = summaries

if "out" in st.session_state:
    out = st.session_state["out"]
    st.subheader("Mood per entry")
    st.dataframe(out[["date", "mood", "score"]])

    st.subheader("Mood trend (1 = low, 5 = high)")
    st.line_chart(out.set_index("date")["score"])

    st.subheader("Weekly summaries")
    for week, (avg, summ) in st.session_state["summaries"].items():
        with st.expander(f"Week {week}  (avg mood {avg:.1f}/5)", expanded=True):
            st.write(summ)
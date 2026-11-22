# PS47 – Daily Journal Summarizer

Tags mood for each journal entry (CSV) using an LLM, then gives weekly summaries and a mood line chart.

**Stack:** Python, Streamlit, Groq API (Llama 3.3 70B), pandas

## Run
```
pip install -r requirements.txt
# create .env with GROQ_API_KEY=your_key
streamlit run app.py
```
CSV format: `date,entry`

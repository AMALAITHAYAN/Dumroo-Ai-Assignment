import os
import json
from datetime import datetime, timedelta

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ---------- CONFIG ----------

DATA_PATH = "data/students.csv"

# Simulated logged-in admin scope (not super admin)
ADMIN_SCOPE = {
    "grade": 8,
    "class_section": "A",
    "region": "North",
}

# Make sure OPENAI_API_KEY is set in your environment
# Make sure OPENAI_API_KEY is set in your environment
GEMINI_MODEL = "gemini-2.0-flash"
GOOGLE_API_KEY = "AIzaSyDi0ijPT8mU3t5drGHvdejGpFBgWY9T49g"




# ---------- DATA LOADING & SCOPE ----------

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["homework_due_date", "quiz_date"])
    return df


def apply_admin_scope(df, scope):
    """Restrict data to the admin's assigned grade/class/region."""
    if scope.get("grade") is not None:
        df = df[df["grade"] == scope["grade"]]
    if scope.get("class_section"):
        df = df[df["class_section"] == scope["class_section"]]
    if scope.get("region"):
        df = df[df["region"] == scope["region"]]
    return df


# ---------- LLM QUERY PARSING ----------
def get_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )



PARSER_SYSTEM_PROMPT = """
You are a JSON API that converts natural language questions from a school admin
into a small JSON object describing how to filter a student dataset.

The dataset has these columns:
- student_id (int)
- student_name (str)
- grade (int)
- class_section (str)
- region (str)
- homework_name (str)
- homework_submitted ("Yes" or "No")
- homework_due_date (date, YYYY-MM-DD)
- quiz_name (str)
- quiz_score (int)
- quiz_date (date, YYYY-MM-DD)

The admin can ask questions like:
- "Which students haven’t submitted their homework yet?"
- "Show me performance data for Grade 8 from last week"
- "List all upcoming quizzes scheduled for next week"

You MUST respond with VALID JSON ONLY, no extra text.

JSON schema:
{{
  "intent": "homework_status" | "performance" | "quizzes" | "unknown",
  "filters": {{
    "grade": "int or null",
    "class_section": "string or null",
    "region": "string or null"
  }},
  "time_range": {{
    "type": "last_week" | "this_week" | "next_week" | "date_range" | null,
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null"
  }}
}}

Rules:
- If the question doesn't specify grade/class/region, set them to null.
- If the question says things like "for Grade 8", "for class A", fill those in.
- For "last week", "this week", "next week" use the `type` field.
- If the question mentions explicit dates, use "date_range" and set start_date/end_date.
- If you don't understand the question, set intent to "unknown".
"""

parser_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PARSER_SYSTEM_PROMPT),
        (
            "user",
            "Admin scope: {admin_scope}\n\nQuestion: {question}\n\nReturn JSON now.",
        ),
    ]
)


def parse_question(question: str, admin_scope: dict) -> dict:
    llm = get_llm()
    chain = parser_prompt | llm

    response = chain.invoke(
        {"question": question, "admin_scope": json.dumps(admin_scope)}
    )
    raw = response.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if model adds text—try to extract JSON
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1:
            parsed = json.loads(raw[first_brace : last_brace + 1])
        else:
            parsed = {
                "intent": "unknown",
                "filters": {"grade": None, "class_section": None, "region": None},
                "time_range": {"type": None, "start_date": None, "end_date": None},
            }

    return parsed


# ---------- FILTER HELPERS ----------

def apply_filters(df, filters: dict, admin_scope: dict):
    """Apply admin scope first, then any explicit filters from the query."""
    df = apply_admin_scope(df, admin_scope)

    # Optional extra filters from question (e.g. grade 8, class A)
    if filters.get("grade") is not None:
        df = df[df["grade"] == filters["grade"]]
    if filters.get("class_section"):
        df = df[df["class_section"] == filters["class_section"]]
    if filters.get("region"):
        df = df[df["region"] == filters["region"]]

    return df


def apply_time_range(df, time_range: dict, date_col: str):
    if not time_range or not time_range.get("type"):
        return df

    today = datetime.today().date()
    ttype = time_range.get("type")
    start_date = None
    end_date = None

    if ttype == "last_week":
        # Last 7 days (excluding today)
        end_date = today - timedelta(days=1)
        start_date = end_date - timedelta(days=6)
    elif ttype == "this_week":
        # From Monday of this week to Sunday
        start_date = today - timedelta(days=today.weekday())
        end_date = start_date + timedelta(days=6)
    elif ttype == "next_week":
        # Next week's Monday to Sunday
        start_date = today + timedelta(days=(7 - today.weekday()))
        end_date = start_date + timedelta(days=6)
    elif ttype == "date_range":
        if time_range.get("start_date"):
            start_date = datetime.strptime(time_range["start_date"], "%Y-%m-%d").date()
        if time_range.get("end_date"):
            end_date = datetime.strptime(time_range["end_date"], "%Y-%m-%d").date()

    if start_date:
        df = df[df[date_col].dt.date >= start_date]
    if end_date:
        df = df[df[date_col].dt.date <= end_date]

    return df


# ---------- INTENT HANDLERS ----------

def handle_homework_status(df):
    # "Which students haven’t submitted their homework yet?"
    not_submitted = df[df["homework_submitted"] == "No"]
    cols = ["student_id", "student_name", "grade", "class_section", "region", "homework_name", "homework_due_date"]
    return not_submitted[cols].sort_values(by="homework_due_date")


def handle_performance(df, time_range):
    # "Show me performance data for Grade 8 from last week"
    df = apply_time_range(df, time_range, date_col="quiz_date")
    cols = ["student_id", "student_name", "grade", "class_section", "region", "quiz_name", "quiz_score", "quiz_date"]
    return df[cols].sort_values(by="quiz_date", ascending=False)


def handle_quizzes(df, time_range):
    # "List all upcoming quizzes scheduled for next week"
    df = apply_time_range(df, time_range, date_col="quiz_date")
    # Show unique quiz events for this scope
    cols = ["quiz_name", "grade", "class_section", "region", "quiz_date"]
    df_small = df[cols].drop_duplicates().sort_values(by="quiz_date")
    return df_small


def answer_question(question: str, admin_scope: dict):
    df = load_data()
    parsed = parse_question(question, admin_scope)

    intent = parsed.get("intent", "unknown")
    filters = parsed.get("filters") or {}
    time_range = parsed.get("time_range") or {}

    scoped_df = apply_filters(df, filters, admin_scope)

    if scoped_df.empty:
        print("\nNo data found in your scope for this query.")
        return

    if intent == "homework_status":
        result = handle_homework_status(scoped_df)
    elif intent == "performance":
        result = handle_performance(scoped_df, time_range)
    elif intent == "quizzes":
        result = handle_quizzes(scoped_df, time_range)
    else:
        print("\nSorry, I couldn't understand that question well enough to answer.")
        print("Try something like:")
        print('- "Which students haven’t submitted their homework yet?"')
        print('- "Show me performance data for Grade 8 from last week"')
        print('- "List all upcoming quizzes scheduled for next week"')
        return

    if result.empty:
        print("\nNo matching rows after filtering.")
    else:
        print("\n--- Answer ---")
        print(result.to_string(index=False))


# ---------- CLI ENTRYPOINT ----------

def main():
    print("Dumroo Admin NLQ Demo")
    print(f"Your admin scope: {ADMIN_SCOPE}")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Ask a question: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        if not q.strip():
            continue

        answer_question(q, ADMIN_SCOPE)
        print("\n")


if __name__ == "__main__":
    main()

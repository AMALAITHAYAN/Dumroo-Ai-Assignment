import os
import json
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

DATA_PATH = "data/students.csv"

# Gemini model + API key (LOCAL ONLY – don't commit real key to GitHub)
GEMINI_MODEL = "gemini-2.0-flash"
GOOGLE_API_KEY = "AIzaSyDi0ijPT8mU3t5drGHvdejGpFBgWY9T49g"  # <- put your key here

# You could also let the admin pick their scope from UI
DEFAULT_ADMIN_SCOPE = {
    "grade": 8,
    "class_section": "A",
    "region": "North",
}

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
        ("user", "Admin scope: {admin_scope}\n\nQuestion: {question}\n\nReturn JSON now."),
    ]
)


def get_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )


def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["homework_due_date", "quiz_date"])


def apply_admin_scope(df, scope):
    if scope.get("grade") is not None:
        df = df[df["grade"] == scope["grade"]]
    if scope.get("class_section"):
        df = df[df["class_section"] == scope["class_section"]]
    if scope.get("region"):
        df = df[df["region"] == scope["region"]]
    return df


def parse_question(question, admin_scope):
    llm = get_llm()
    chain = parser_prompt | llm
    resp = chain.invoke(
        {"question": question, "admin_scope": json.dumps(admin_scope)}
    )
    raw = resp.content.strip()
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1:
        return json.loads(raw[first_brace : last_brace + 1])
    return {
        "intent": "unknown",
        "filters": {"grade": None, "class_section": None, "region": None},
        "time_range": {"type": None, "start_date": None, "end_date": None},
    }


def apply_time_range(df, time_range, date_col):
    if not time_range or not time_range.get("type"):
        return df

    today = datetime.today().date()
    ttype = time_range.get("type")
    start_date = None
    end_date = None

    if ttype == "last_week":
        end_date = today - timedelta(days=1)
        start_date = end_date - timedelta(days=6)
    elif ttype == "this_week":
        start_date = today - timedelta(days=today.weekday())
        end_date = start_date + timedelta(days=6)
    elif ttype == "next_week":
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


def handle_homework_status(df):
    not_submitted = df[df["homework_submitted"] == "No"]
    cols = [
        "student_id",
        "student_name",
        "grade",
        "class_section",
        "region",
        "homework_name",
        "homework_due_date",
    ]
    return not_submitted[cols].sort_values(by="homework_due_date")


def handle_performance(df, time_range):
    df = apply_time_range(df, time_range, "quiz_date")
    cols = [
        "student_id",
        "student_name",
        "grade",
        "class_section",
        "region",
        "quiz_name",
        "quiz_score",
        "quiz_date",
    ]
    return df[cols].sort_values(by="quiz_date", ascending=False)


def handle_quizzes(df, time_range):
    df = apply_time_range(df, time_range, "quiz_date")
    cols = ["quiz_name", "grade", "class_section", "region", "quiz_date"]
    return df[cols].drop_duplicates().sort_values(by="quiz_date")


def main():
    st.title("Dumroo Admin NLQ Demo")

    with st.sidebar:
        st.header("Admin Scope")
        grade = st.number_input(
            "Grade",
            min_value=1,
            max_value=12,
            value=DEFAULT_ADMIN_SCOPE["grade"],
        )
        class_section = st.text_input(
            "Class Section", value=DEFAULT_ADMIN_SCOPE["class_section"]
        )
        region = st.text_input("Region", value=DEFAULT_ADMIN_SCOPE["region"])
        admin_scope = {
            "grade": int(grade),
            "class_section": class_section.strip(),
            "region": region.strip(),
        }

    question = st.text_input(
        "Ask a question",
        placeholder="e.g. Which students haven’t submitted their homework yet?",
    )

    if st.button("Ask") and question.strip():
        df = load_data()
        df = apply_admin_scope(df, admin_scope)
        parsed = parse_question(question, admin_scope)

        intent = parsed.get("intent", "unknown")
        filters = parsed.get("filters") or {}
        time_range = parsed.get("time_range") or {}

        # Apply extra filters from question if any
        if filters.get("grade") is not None:
            df = df[df["grade"] == filters["grade"]]
        if filters.get("class_section"):
            df = df[df["class_section"] == filters["class_section"]]
        if filters.get("region"):
            df = df[df["region"] == filters["region"]]

        if df.empty:
            st.warning("No data found in your scope for this query.")
            return

        if intent == "homework_status":
            result = handle_homework_status(df)
        elif intent == "performance":
            result = handle_performance(df, time_range)
        elif intent == "quizzes":
            result = handle_quizzes(df, time_range)
        else:
            st.error(
                "I couldn't understand that question. Try one of the example queries."
            )
            return

        if result.empty:
            st.info("No rows matched your filters.")
        else:
            st.subheader("Answer")
            st.dataframe(result, use_container_width=True)


if __name__ == "__main__":
    main()

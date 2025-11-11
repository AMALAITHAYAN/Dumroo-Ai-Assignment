> ⚠️ API key note  
> This project currently uses a **dummy Google Gemini API key** from a throwaway Google account just for demo/learning purposes.  
> **In any real project, never expose real API keys in public repos.** Always keep them in environment variables or a `.env` file.

# Dumroo Admin NLQ Assignment

This repo implements an AI-powered natural language querying feature for a hypothetical Dumroo Admin Panel. Admins can ask simple English questions like:

- "Which students haven’t submitted their homework yet?"
- "Show me performance data for Grade 8 from last week"
- "List all upcoming quizzes scheduled for next week"

The system converts these questions into structured filters, applies role-based scoping, and returns filtered data from a small CSV dataset.

## Tech Stack

- Python
- Pandas
- LangChain + Gemini API
- Streamlit for a simple UI

## Dataset

The sample dataset is in `data/students.csv` with fields:

- `student_id`
- `student_name`
- `grade`
- `class_section`
- `region`
- `homework_name`
- `homework_submitted`
- `homework_due_date`
- `quiz_name`
- `quiz_score`
- `quiz_date`

## Role-Based Access

Each admin has a scope defined by:

- `grade`
- `class_section`
- `region`

All queries are automatically restricted to the admin’s scope before any other filters, so admins cannot see platform-wide data or other grades/regions.

Scope is configured in `app.py` as:

```python
ADMIN_SCOPE = {
    "grade": 8,
    "class_section": "A",
    "region": "North",
}



How to run this project
1. Prerequisites
Python 3.10+

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt





You’ll see:

Dumroo Admin NLQ Demo
Your admin scope: {'grade': 8, 'class_section': 'A', 'region': 'North'}
Type 'exit' to quit.


Now type questions like:

Which students haven’t submitted their homework yet?

Show me performance data for Grade 8 from last week

List all upcoming quizzes scheduled for next week

Type exit (or quit) to close the program.




Running the Streamlit UI

This is a simple web UI for the same functionality.

source .venv/bin/activate      # if not already active
streamlit run streamlit_app.py


Then open the URL shown in the terminal (usually http://localhost:8501).


Type a natural language question:

e.g. Which students haven’t submitted their homework yet?

Click Ask to see the filtered table of results.

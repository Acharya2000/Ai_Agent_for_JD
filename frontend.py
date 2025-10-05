import streamlit as st
import requests

st.title("Job Description Generator & Candidate Selector")

# ---------- Inputs ----------
api_key = st.text_input("OpenAI API Key", type="password")

topic = st.text_area(
    "Job Description Topic",
    "generate Job description for my company name Laxmi chect fund, For this topic Data Science, with required skills: Python, ML, MLOps, DL"
)

iteration = st.number_input("Iteration", min_value=0, value=0)
min_no_cv = st.number_input("Minimum CVs required", min_value=1, value=1)

# Interview date and time
interview_date = st.date_input("Select Interview Date")
interview_time = st.time_input("Select Interview Time")

# ---------- Submit ----------
if st.button("Generate JD and Select Candidates"):

    if not api_key:
        st.error("Please provide your OpenAI API Key!")
    else:
        # Prepare payload
        payload = {
            "api_key": api_key,
            "topic": topic,
            "iteration": iteration,
            "min_no_cv_you_want": min_no_cv,
            "interview_date": str(interview_date),
            "interview_time": interview_time.strftime("%H:%M")
        }

        # Call FastAPI endpoint
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            result = response.json()["result"]

            # ---------- Display Results ----------
            st.subheader("Generated Job Description")
            st.write(result.get("tweet", "JD not generated"))

            st.subheader("Selected Candidates")
            for candidate in result.get("selected_student_for_interview", []):
                st.write(f"Name: {candidate['name']}, Email: {candidate['email']}, Phone: {candidate['phone']}")

            st.subheader("Generated Emails for Interview")
            for mail_data in result.get("mail_generated_for_selected_students", []):
                st.write(f"Mail: {mail_data['mail']}")
                st.write(f"Interview Date & Time: {mail_data['date']}")

        except Exception as e:
            st.error(f"Error calling API: {e}")

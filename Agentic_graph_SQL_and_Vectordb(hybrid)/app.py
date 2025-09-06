# app.py (Streamlit UI for your FastAPI backend)

import streamlit as st
import requests

# Base URL of your FastAPI backend
BASE_URL = "http://127.0.0.1:8000"

# Streamlit page setup
st.set_page_config(page_title="Agentic RAG System", page_icon="🤖", layout="centered")

st.title("🤖 Agentic RAG System")
st.write("This is a Streamlit UI for your FastAPI-powered Agentic workflow.")

# ---------------- LOGIN ----------------
st.subheader("🔑 Login")

with st.form("login_form"):
    uname = st.text_input("Username")
    pswd = st.text_input("Password", type="password")
    login_btn = st.form_submit_button("Login")

if login_btn:
    response = requests.post(f"{BASE_URL}/login", json={"uname": uname, "pswd": pswd})
    if response.status_code == 200:
        result = response.json()
        st.session_state["login_status"] = result["message"]
        st.success(result["message"])
    else:
        st.error("Login failed. Please try again.")

# ---------------- QUERY ----------------
if st.session_state.get("login_status") == "Login Successfull":
    st.subheader("💬 Ask a Question")

    with st.form("query_form"):
        user_query = st.text_area("Enter your question")
        query_btn = st.form_submit_button("Submit")

    if query_btn:
        if not user_query.strip():
            st.warning("Query cannot be empty.")
        else:
            response = requests.post(f"{BASE_URL}/query", json={"query": user_query})
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Answer received!")
                st.write(result.get("Answer", "No answer found."))
            else:
                st.error("Error occurred while processing the query.")
else:
    st.info("Please log in to continue.")

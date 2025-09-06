# app.py - Streamlit frontend for FastAPI

import streamlit as st
import requests

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="LLM SQL Agent", layout="wide")
st.title("💬 LLM-Powered SQL Agent (via FastAPI)")

API_URL = "http://127.0.0.1:8000"   # FastAPI backend URL

# Sidebar login
st.sidebar.header("🔑 Login")
user_id = st.sidebar.text_input("User ID")
user_pwd = st.sidebar.text_input("Password", type="password")
login_btn = st.sidebar.button("Login")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if login_btn:
    try:
        response = requests.post(
            f"{API_URL}/login",
            json={"user_id": user_id, "usr_pwd": user_pwd}
        )
        if response.status_code == 200 and "successfully" in response.text:
            st.session_state.logged_in = True
            st.sidebar.success("✅ You are logged in successfully")
        else:
            st.session_state.logged_in = False
            st.sidebar.error("❌ Wrong credentials")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error connecting to API: {e}")

# Main query interface
if st.session_state.logged_in:
    st.subheader("Ask your database a question")
    user_query = st.text_area("Enter your query:", height=100)

    if st.button("Run Query"):
        try:
            response = requests.post(
                f"{API_URL}/User_input",
                json={"query": user_query}
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    st.success("✅ Query executed successfully")
                    st.write("**Answer:**", result["results"]["content"])
                else:
                    st.error(f"❌ {result.get('error', 'Unknown error')}")
            else:
                st.error(f"❌ API Error: {response.status_code}")
        except Exception as e:
            st.error(f"⚠️ Error connecting to API: {e}")
else:
    st.info("🔒 Please login to continue.")

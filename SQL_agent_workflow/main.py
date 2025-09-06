from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from sql_prompt import sql_system_prompt 
from SQL_agentic_graph_workflow import sql_agentic_workflow
from dotenv import load_dotenv

import logging
import os
from pydantic import BaseModel
from fastapi import FastAPI

# configure logging
logging.basicConfig(filename="app.log",
                    level = logging.ERROR,
                    format = "%(asctime)s - %(levelname)s - %(message)s")


# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("openai_api_key not found in environment variables")
    raise ValueError("Mission openai_api_key. Please check the .env file.")

# Initialize LLM
llm = init_chat_model('gpt-4o-mini', model_provider = "openai")

db_path ="sqlite:///Chinook.db"

app =FastAPI()



# Database integration
def sql_db():
    """ Connect to sqlite database"""
    try:
        
        db = SQLDatabase.from_uri(db_path)
        return db

    except Exception  as e:
        logging.error(f"Database connection failed :{db_path}")
        print(f"Database connection failed {e}")
        return None
class Login(BaseModel):
    user_id:     str
    usr_pwd:    str

class Uinput(BaseModel):
    query:  str

# LLM agent query
@app.get("/")
def home_page():
    return "Wecome to home page"

@app.post("/login")
def user_login(login: Login)-> str:
    try:
        result = (
            "You are logged in successfully" if login.user_id == '1234' and login.usr_pwd =='1234' else "your credential is wrong"
        )
        logging.info(f"Login Result: {result}") 
        return result 
    except Exception as e :
      logging.info(f"your credential is wrong: (str{e})")
      return {"Error:": " An unexpected error occures during login."}
  
@app.post("/User_input")
def user_input(uinput: Uinput) ->dict:
    user_query = uinput.query
    try:
        result  = sql_agentic_workflow(db_path, user_query)
        print("Final output:", result)
        
        if 'messages' not in result or not result["messages"]:
            logging.error("Agent retuned empty response")
            
        return {"success": True, "results": result}


    except Exception as e :
        logging.error(" Unexpected error occured while llm invoke.")
        print(f"Unexpected error occured while llm invoke: {e}")
        return {"success": False, "error": str(e)}


# ---------------- Entry Point ----------------
if __name__ == "__main__":
    import uvicorn
    user_query = "List the total number of employees in each city"
    db_path ="sqlite:///Chinook.db"

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
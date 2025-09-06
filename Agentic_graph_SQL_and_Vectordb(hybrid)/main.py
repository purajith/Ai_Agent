import logging
from pydantic import BaseModel
from fastapi import FastAPI
from agent_workflow_graph_hybrid import Agentic_flow
import uvicorn


# ----------------- LOGGING CONFIGURATION -----------------
logging.basicConfig(
    filename="logs/app.log",  # Log file path
    level=logging.INFO,       # Logging level (DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)
pdf_path = "data/THERMAL-ENGINEERING-2.pdf"
db_path ="sqlite:///Chinook.db"  

class Ulogin(BaseModel):
    uname: str
    pswd: str
    
class UserQuery(BaseModel):
    query: str
    
app = FastAPI()

@app.get("/")
def home_page():
    return{"message":"welcome to the Agentic Rag system"}

@app.post("/login")
def User_login(ulogin:Ulogin)->dict[str,str]:
    try: 
            
        if ulogin.uname =="admin" and ulogin.pswd =="admin123":
            return {"message":"Login Successfull"}
        else:
            return {"message":"Invalid Credentials"}
    except Exception as e:
        logging.info(f" Error occured during login:{str(e)} ")
        return {"message":"Error during login"}
class QueryResponse(BaseModel):
    Answer: dict  # or more detailed schema if you know structure

@app.post("/query", response_model=QueryResponse)
def User_query(uquery: UserQuery):
    try:
        if uquery.query=="":
            return {"Answer": {"message": "Query cannot be empty"}}
        
        question = uquery.query
        result= Agentic_flow(question,db_path,pdf_path)
        return {"Answer":result} 
    except Exception as e:
        logging.info(f" Error occured during query processing:{str(e)} ")
        return {"message":"Error during query processing"}

    
    
if __name__ =="__main__":
  
    uvicorn.run("agent:app", host="127.0.0.1", port=8000, reload=True)

# ----------------- IGNORE -----------------    

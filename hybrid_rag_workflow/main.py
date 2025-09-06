import logging
from pydantic import BaseModel
from fastapi import FastAPI
from agent_workflow_graph_hybrid import Agentic_flow


# ----------------- LOGGING CONFIGURATION -----------------
logging.basicConfig(
    filename="logs/app.log",  # Log file path
    level=logging.INFO,       # Logging level (DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

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

@app.post("/query")
def User_query(uquery:UserQuery)->dict[str,str]:
    try:
        if uquery.query=="":
            return {"message":"Query cannot be empty"}
        
        question = uquery.query
        result = Agentic_flow(question)
        return {"Answer":result} 
    except Exception as e:
        logging.info(f" Error occured during query processing:{str(e)} ")
        return {"message":"Error during query processing"}

    
    
if __name__ =="__main__":
    import uvicorn 
    uvicorn.run("agent:app", host="127.0.0.1", port=8000, reload=True)

# ----------------- IGNORE -----------------    
    
    
    
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from sql_prompt import sql_system_prompt 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing import Annotated, Literal
from typing_extensions import TypedDict, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import logging
import os
from langchain.tools import tool



load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
memory = MemorySaver()

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


class State(TypedDict):
    query: str
    query_rewrite: Dict
    result: Dict
    plot_result: str
    db_path: str

def prompt_plot_agent(result) -> str:
    prompt = """Act as a plotting agent. With SQL data `df`:
    - If suitable for bar/pie/hist, output ONLY runnable matplotlib code making ONE plot (no seaborn, no subplots, no colors, handle NaNs, aggregate as needed, top 10 categories, rotate ticks if long, add labels+title). 
    - Else output EXACTLY: answer: <why not suitable>.
    Rules: 
    - Bar/Pie: one low-cardinality categorical (≤20 unique) + one numeric (aggregate). Pie only if ≤8 categories. 
    - Hist: one numeric column. 
    - No plots for high-cardinality categoricals, text, dates, or multiple columns.
    Output: code only OR the single-line “answer: …”.
    """
    
    prompt_plot = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Here is the SQL result data: {result}"}
    ]
    return prompt_plot

# Database integration
def tool_sql_db(db_path):
    """ Connect to sqlite database"""
    try:
        db = SQLDatabase.from_uri(db_path)
        
        if db is None:
            return {"success": False, "error": "Database connection failed"}
        
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
        return tools

    except Exception  as e:
        logging.error(f"Database tools creation failed :{db_path}")
        print(f"Database tools creation failed {e}")
        return None
    
def query_rewrite(state: State) -> Dict:
    try:
        query = state["query"]
        print("Original question:", query)
        if not query:
            logging.error("Input query is empty")
            raise ValueError("Input query is empty")
        prompt =f" Rewrite the following question to be more precise and and understandable if necessary: {query}"
        response = llm.invoke( prompt)
        return {"query_rewrite":response.content }
    
    except Exception as e:
        logging.info(f"Query rewrite failed: {e}")
        return {"query_rewrite": "Query rewrite failed"}  
      
def sql_agent(state: State) ->dict:
    question = state["query_rewrite"]
    if not question:
        logging.error("Rewritten query is empty")
        raise ValueError("Rewriten query is empty")
    
    print("Rewritten question:", question)
    try:
        tools  = tool_sql_db(db_path)
        if tools is None:
            return {"Success":False, "error": "Tool creation failed"}
        
        agent_executor = create_react_agent(llm,tools, prompt = sql_system_prompt())
        
        result =agent_executor.invoke(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="values",
            )
        
        if 'messages' not in result or not result["messages"]:
            logging.error("Agent retuned empty response")
            
        return {"result":result["messages"][-1]}

    except Exception as e :
        logging.error(" Unexpected error occured while llm invoke.")
        print(f"Unexpected error occured while llm invoke: {e}")
        return {"success": False, "error": str(e)}

def plotting_agent(state:State) -> str:
    try:
        if state["result"] is None:
            logging.error("NO result to plot")
            raise ValueError("No result to plot")
        result = state["result"]
        prompt = prompt_plot_agent(result)
        response = llm.invoke(content = prompt)
        return {"plot_result":response.content }
    
    except Exception as e:
        logging.info(f"Plotting agent failed: {e}")
        return {"plot_result": "Plotting_failed"}
    



def sql_agentic_workflow(db_path:str, query:str ) -> dict:
    query = query
    if query is None or query.strip() =="":
        logging.error("Input query is empty")
        raise ValueError("Input query is empty")
    
    tools = tool_sql_db(db_path)   # ✅ now using db_path
    if tools is None:
            return {"Success":False, "error": "Tool creation failed"}
    builder =StateGraph(State)
    # add nodes
    builder.add_node("query_rewrite_node", query_rewrite)
    builder.add_node("sql_agent_node", sql_agent)
    builder.add_node("plotting_agent_node", plotting_agent)

    # add edges
    builder.add_edge(START,"query_rewrite_node")
    builder.add_edge("query_rewrite_node","sql_agent_node")
    builder.add_edge("sql_agent_node", "plotting_agent_node") 
    builder.add_edge("plotting_agent_node",END)
    
    graph = builder.compile()
    result = graph.invoke({"query": query})

    return result


# db_path ="sqlite:///Chinook.db"
# user_query = "List the total number of employees in each city"
# output  = sql_agentic_workflow(db_path, user_query)
# print("Final output:", output)



# compile graph
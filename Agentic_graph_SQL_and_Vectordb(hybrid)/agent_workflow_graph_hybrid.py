from retrieval import load_embedings, embeding_save_local
from typing_extensions import TypedDict, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from typing import Annotated, Literal
from langchain_community.utilities import SQLDatabase
from sql_prompt import sql_system_prompt 
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import logging
import os




load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
memory = MemorySaver()


# ----------------- LOGGING CONFIGURATION -----------------
logging.basicConfig(
    filename="logs/app.log",  # Log file path
    level=logging.INFO,       # Logging level (DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

# API 
load_dotenv()
# Initialize LLM
llm = init_chat_model('gpt-4o-mini', model_provider = "openai")
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("openai_api_key not found in environment variables")
    raise ValueError("Mission openai_api_key. Please check the .env file.")

# ---------- One-time build ----------
pdf_path = "data/THERMAL-ENGINEERING-2.pdf"
db_path ="sqlite:///Chinook.db"

#-------
def get_hybrid_retrieval(pdf_path):
    try:
        retriever = load_embedings()

        if retriever ==None:
            embeding_save_local(pdf_path)
            retriever = load_embedings()
        return retriever
    except Exception as e:
        logging.info(f" Error occured while loading the Embedings:{str(e)} ")
        return "Error in embedings"
    

class State(TypedDict):

    query: str
    query_rewrite:   str
    hybrid_retrieved_data: str
    hybrid_llm_result:  str
    sql_result: str
    # plot_result: str
    sql_retrieved_data: str

    
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
        # print("sql_agent_result",result)
        if 'messages' not in result or not result["messages"]:
            logging.error("Agent retuned empty response")
            
        return {"sql_result":result}
    except Exception as e :
        logging.error(" Unexpected error occured while llm invoke.")
        print(f"Unexpected error occured while llm invoke: {e}")
        return {"success": False, "error": str(e)}

# def plotting_agent(state:State) -> str:
#     try:
#         if state["result"] is None:
#             logging.error("NO result to plot")
#             raise ValueError("No result to plot")
#         result = state["result"]
#         prompt = prompt_plot_agent(result)
#         response = llm.invoke(prompt)
#         return {"plot_result":response.content }
    
#     except Exception as e:
#         logging.info(f"Plotting agent failed: {e}")
#         return {"plot_result": "Plotting_failed"}
    

def rewrite_llm(state: State) ->  Dict[str, str]:
    # Extract query from state
    query = state["query"]

     # System + User messages
    prompt = [
        {"role": "system", "content": "You are a helpful assistant that rewrites user queries for clarity. If a query is already clear and well-formed, do not rewrite it — return it exactly as it is."},
        {"role": "user", "content": query}
    ]
    # Call LLM
    response = llm.invoke(prompt)

    return {"query_rewrite": response.content}


def hybrid_retrieval(state:State)-> str:
    # --- Query ---
    query_rewrite = state["query_rewrite"]
    # retrieval bm25 + faiss
    retriever = get_hybrid_retrieval(pdf_path)
    results = retriever.get_relevant_documents(query_rewrite)
    return {'hybrid_retrieved_data':"\n\n".join([doc.page_content for doc in results])}


def hybrid_llm_result(state:State)->str:
    hybrid_retrieved_data = state['hybrid_retrieved_data']
    re_writen_query = state["query_rewrite"]
    prompt = [
    {"role": "system", "content": "You are a knowledgeable assistant. Use the retrieved context to answer."},
    {"role": "user", "content": f"Question: {re_writen_query}\n\nContext:\n{hybrid_retrieved_data}\n\nAnswer:"}
]

    response = llm.invoke(prompt)
    return {'hybrid_llm_result': response.content}


def decision_model(state: State)-> Literal["sql_retrieval_node", "hybrid_retrieval_node"]:
         # System + User messages
    prompt = [
        {"role": "system", "content": "You are a helpful assistant decission make assistant in agentic rag. if the user query is related to thermal engineering you can  return the result 'hybrid' else pass 'sql'"},
        {"role": "user", "content": f' Youser query is :{state["query"]}'}
    ]
    # Call LLM
    response = llm.invoke(prompt)
    print(f"Decision model response: {response.content}")

    if response.content =='sql':
        return "sql_retrieval_node"
    elif response.content =='hybrid':
        return "hybrid_retrieval_node"
# separate chooser
def Agentic_flow(question,db_path,pdf_path)-> str:
    pdf_path = pdf_path
    if pdf_path is None or pdf_path.strip() =="" :
        logging.error("PDF path is empty")
        raise ValueError("PDF path is empty")
    
    query = question
    if query is None or query.strip() =="":
        logging.error("Input query is empty")
        raise ValueError("Input query is empty")
    
    tools = tool_sql_db(db_path)   # ✅ now using db_path
    if tools is None:
            return {"Success":False, "error": "Tool creation failed"}
    try:
        
        builder = StateGraph(State)
        builder.add_node('rewrite_llm_node', rewrite_llm)
        builder.add_node('hybrid_retrieval_node', hybrid_retrieval)
        builder.add_node("sql_retrieval_node",sql_agent)
        builder.add_node("hybrid_llm_result_node", hybrid_llm_result)
        # builder.add_node("Sql_ploting", plotting_agent)

        #builder.add_node("decision_model",decision_model)

        builder.add_edge(START, 'rewrite_llm_node')
        builder.add_conditional_edges("rewrite_llm_node",decision_model)
        builder.add_edge("hybrid_retrieval_node", "hybrid_llm_result_node")
        # builder.add_edge("sql_retrieval_node", "Sql_ploting")

        builder.add_edge("hybrid_llm_result_node", END)
        builder.add_edge("sql_retrieval_node", END)

        #builder.add_edge("hybrid_retrieval_node","sql_retrieval_node")
        graph = builder.compile()
        graph
        result = graph.invoke({"query":query})
    except Exception as e:
        logging.error(" Error in Agentic_flow execution")
        print(f"Error in Agentic_flow execution: {e}")
    return result
    # print(result)



# if __name__ == "__main__":
#     question = "Whtat is thermal engineering?"
#     result= Agentic_flow(question,db_path,pdf_path)
#     print("Agentic flow completed.", result)













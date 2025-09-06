from retrieval import load_embedings, embeding_save_local
from langchain.chat_models import init_chat_model
import logging
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
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
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = init_chat_model('gpt-4o-mini')

# ---------- One-time build ----------
pdf_path = "data/THERMAL-ENGINEERING-2.pdf"

#-------
def get_hybrid_retrieval():
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
    rewrite_query:   str
    hybrid_retrieved_data: str
    hybrid_llm_result:  str
    sql_retrieved_data: str

    
    


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

    return {"rewrite_query": response.content}


def hybrid_retrieval(state:State)-> str:
    # --- Query ---
    rewrite_query = state["rewrite_query"]
    # retrieval bm25 + faiss
    retriever = get_hybrid_retrieval()
    results = retriever.get_relevant_documents(rewrite_query)
    return {'hybrid_retrieved_data':"\n\n".join([doc.page_content for doc in results])}


def hybrid_llm_result(state:State)->str:
    hybrid_retrieved_data = state['hybrid_retrieved_data']
    re_writen_query = state["rewrite_query"]
    prompt = [
    {"role": "system", "content": "You are a knowledgeable assistant. Use the retrieved context to answer."},
    {"role": "user", "content": f"Question: {re_writen_query}\n\nContext:\n{hybrid_retrieved_data}\n\nAnswer:"}
]

    response = llm.invoke(prompt)
    return {'hybrid_llm_result': response.content}

# separate chooser
def Agentic_flow(question):
    builder = StateGraph(State)
    builder.add_node('rewrite_llm_node', rewrite_llm)
    builder.add_node('hybrid_retrieval_node', hybrid_retrieval)
    builder.add_node("hybrid_llm_result_node", hybrid_llm_result)

    builder.add_edge(START, 'rewrite_llm_node')
    builder.add_edge("rewrite_llm_node","hybrid_retrieval_node")

    builder.add_edge("hybrid_retrieval_node","hybrid_llm_result_node")
    # builder.add_edge(['hybrid_retrieval_node','sql_retrieval_node'], END)
    builder.add_edge("hybrid_llm_result_node", END)

    graph = builder.compile()
    graph

    result = graph.invoke({"query":question})
    print(result["hybrid_llm_result"])
    return result["hybrid_llm_result"] 


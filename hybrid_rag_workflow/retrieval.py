from flashrank import Ranker  # Must come first!
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from flashrank import Ranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from data_extraction import extract_text_from_pdf, chunk_text_token_based,to_documents
import os 


# re ranking model
flashrank_model = "ms-marco-MiniLM-L-12-v2"  
ranker = Ranker(model_name=flashrank_model)
reranker = FlashrankRerank(client=ranker, model=flashrank_model, top_n=6)



# ---------- One-time build ----------

# to save the extracted data in local:
def embeding_save_local(pdf_path):
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text_token_based(raw_text, max_tokens=500, overlap=50)
    docs = to_documents(chunks, source=pdf_path)

    # --- BM25 retriever ---
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 8

    with open("embeding_models/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # --- FAISS vector db ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    vectordb.save_local("embeding_models/faiss_index")

# --------------
def load_embedings():
    try:
        # --- Load BM25 ---

        with open("embeding_models/bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        bm25.k = 8
        # --- Load FAISS ---
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.load_local("embeding_models/faiss_index", embeddings, allow_dangerous_deserialization=True)
        faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    except Exception as e:
        print(f"An Error in load embeding pkl file of bm25 & faiss")
        return None 
    # --- Hybrid retriever ---
    hybrid = EnsembleRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[0.5, 0.5]
    )

    # --- Reranker ---
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    reranker = FlashrankRerank(client=ranker, model="ms-marco-MiniLM-L-12-v2", top_n=6)
    retriever = ContextualCompressionRetriever(base_retriever=hybrid, base_compressor=reranker)
    return retriever

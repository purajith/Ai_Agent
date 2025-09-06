from langchain_core.documents import Document
from typing import List

import pdfplumber
import tiktoken

# ---------- PDF extract + chunk ----------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def chunk_text_token_based(text: str, tokenizer_model: str = "gpt-4o-mini",
                           max_tokens: int = 500, overlap: int = 50) -> List[str]:
    enc = tiktoken.encoding_for_model(tokenizer_model)
    tokens = enc.encode(text)
    chunks, start = [], 0
    stride = max_tokens - overlap
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens): break
        start += stride
    return chunks

def to_documents(chunks: List[str], source: str = "pdf") -> List[Document]:
    return [Document(page_content=ch, metadata={"source": source, "chunk_id": i}) for i, ch in enumerate(chunks)]

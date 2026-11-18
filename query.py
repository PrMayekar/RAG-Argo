# NETCDF-FILES/query.py  — full replacement (Ollama → Groq)

import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ================================
# CONFIG
# ================================
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "argo_profiles"

EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================================
# GET CHROMA COLLECTION
# ================================
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDING_FN
    )

# ================================
# SEARCH CHUNKS
# ================================
def search_chunks(question: str, collection, top_k: int = 5):
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    return chunks, metadatas

# ================================
# BUILD PROMPT
# ================================
def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(
        [f"Profile {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)]
    )
    return f"""You are an oceanography assistant.
Answer using ONLY the Argo float profile data below.
Provide exact numeric values — latitude, longitude, cycle number, date, temperature, salinity.
Do NOT guess or make assumptions. Only use what is in the data.

--- DATA ---
{context}
--- END ---

Question: {question}
Answer:"""

# ================================
# CALL GROQ
# ================================
def ask_groq(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content

# ================================
# MAIN ASK FUNCTION
# ================================
def ask(question: str) -> str:
    collection        = get_collection()
    chunks, metadatas = search_chunks(question, collection, top_k=5)
    prompt            = build_prompt(question, chunks)
    answer            = ask_groq(prompt)
    return answer

# ================================
# TERMINAL MODE (optional)
# ================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Argo Float RAG — powered by Groq LLaMA-3.3 70B")
    print("=" * 55)

    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if not question:
                continue
            print("\n[ANSWER]")
            print(ask(question))
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
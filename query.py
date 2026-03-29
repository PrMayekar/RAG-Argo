import chromadb
from chromadb.utils import embedding_functions
import ollama

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "argo_profiles"

EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(
        name               = COLLECTION_NAME,
        embedding_function = EMBEDDING_FN,
    )

def search_chunks(question: str, collection, top_k: int = 3) -> tuple:
    results = collection.query(
        query_texts = [question],
        n_results   = top_k,
    )
    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]

    print(f"\n[SEARCH] Top {top_k} matching profiles:")
    for i, meta in enumerate(metadatas):
        print(f"  {i+1}. Float {meta['wmo_id']} | "
              f"Cycle {meta['cycle_number']} | "
              f"Date {meta['date']}")

    return chunks, metadatas

def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join([f"Profile {i+1}:\n{chunk}"
                            for i, chunk in enumerate(chunks)])
    return f"""You are an oceanography assistant.
Answer using ONLY the Argo float profile data below.
If the answer is not in the data, say "I don't have that information."
Be concise and factual.

--- DATA ---
{context}
--- END ---

Question: {question}
Answer:"""

def ask_llm(prompt: str) -> str:
    response = ollama.chat(
        model    = "phi3:mini",   # lightweight, 8GB RAM friendly
        messages = [{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def ask(question: str):
    print(f"\n{'='*55}")
    print(f"Question: {question}")
    print('='*55)

    collection            = get_collection()
    chunks, metadatas     = search_chunks(question, collection, top_k=3)
    prompt                = build_prompt(question, chunks)

    print("\n[LLM] Generating answer...")
    answer = ask_llm(prompt)

    print(f"\n[ANSWER]\n{answer}")
    print('='*55)

if __name__ == "__main__":
    print("="*55)
    print("  Argo Float RAG System — powered by Phi-3 Mini")
    print("="*55)
    print("Type your question and press Enter.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if not question:
                continue
            ask(question)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
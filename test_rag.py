

# test_rag.py
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "argo_profiles"

EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=EMBEDDING_FN
)

def test_collection_not_empty():
    count = collection.count()
    assert count > 0, "FAIL: ChromaDB collection is empty!"
    print(f"PASS: ChromaDB has {count} chunks")

def test_query_returns_results():
    results = collection.query(
        query_texts=["temperature at float 5907176"],
        n_results=3
    )
    assert len(results["documents"][0]) == 3, "FAIL: Did not return 3 results!"
    print("PASS: Query returns 3 results")

def test_float_specific_query():
    """When asking about a specific float, that float should appear in top results."""
    results = collection.query(
        query_texts=["float 5907176 temperature salinity"],
        n_results=3
    )
    metadatas = results["metadatas"][0]
    wmo_ids   = [m["wmo_id"] for m in metadatas]
    assert "5907176" in wmo_ids, \
        f"FAIL: Float 5907176 not in top 3 results! Got: {wmo_ids}"
    print(f"PASS: Float 5907176 found in top results {wmo_ids}")

def test_different_queries_give_different_results():
    """Two very different questions should return different chunks."""
    r1 = collection.query(query_texts=["float 5907176"], n_results=1)
    r2 = collection.query(query_texts=["float 1901771"], n_results=1)
    id1 = r1["ids"][0][0]
    id2 = r2["ids"][0][0]
    assert id1 != id2, "FAIL: Different queries returned same result!"
    print(f"PASS: Different queries return different results ({id1} vs {id2})")

def test_metadata_present():
    """Every chunk should have required metadata fields."""
    results = collection.query(
        query_texts=["ocean temperature"],
        n_results=5
    )
    required = ["wmo_id", "cycle_number", "date", "latitude", "longitude"]
    for meta in results["metadatas"][0]:
        for field in required:
            assert field in meta, f"FAIL: Missing metadata field '{field}'"
    print("PASS: All metadata fields present in results")

def test_relevance_score():
    """Top result should be more similar than last result."""
    results = collection.query(
        query_texts=["temperature at float 5907176"],
        n_results=5,
        include=["distances"]
    )
    distances = results["distances"][0]
    assert distances[0] <= distances[-1], \
        "FAIL: Results not ordered by similarity!"
    print(f"PASS: Results ordered by similarity "
          f"(best: {distances[0]:.3f}, worst: {distances[-1]:.3f})")

if __name__ == "__main__":
    print("\n===== RAG RETRIEVAL TESTS =====\n")
    tests = [
        test_collection_not_empty,
        test_query_returns_results,
        test_float_specific_query,
        test_different_queries_give_different_results,
        test_metadata_present,
        test_relevance_score,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR in {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*35}")
    print(f"Results: {passed} passed, {failed} failed")

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.hybrid_retriever import HybridRetriever
from rag.vector_store import VectorStore

class MockVectorStore:
    def search(self, query, top_k):
        # Mock ChromaDB-style output: dict with lists of list
        print(f"[MockVectorStore] Searching for: {query}")
        return {
            'documents': [[
                'Meaning of life is 42',
                'Python is a programming language'
            ]],
            'metadatas': [[
                {'chunk_id': 100, 'filename': 'doc1'},
                {'chunk_id': 101, 'filename': 'doc2'}
            ]],
            'distances': [[0.2, 0.8]]
        }

    def get_document_count(self):
        return 2

def test_hybrid_search():
    print("Testing Hybrid Search...")
    
    # 1. Initialize
    vector_store = MockVectorStore()
    retriever = HybridRetriever(vector_store)
    
    # 2. Add Documents (for BM25)
    chunks = [
        {'text': 'Apple banana cherry', 'metadata': {'chunk_id': 0, 'filename': 'fruits.txt'}},
        {'text': 'Dog cat mouse', 'metadata': {'chunk_id': 1, 'filename': 'animals.txt'}},
        {'text': 'SpecialKey123 is the secret code', 'metadata': {'chunk_id': 2, 'filename': 'secret.txt'}}
    ]
    retriever.add_documents(chunks)
    
    # 3. Search - Keyword (BM25 preferred)
    print("\n[Test 1] Keyword Search (SpecialKey123)")
    results, stats = retriever.search("SpecialKey123", top_k=3)
    
    print(f"Results: {len(results)}")
    for i, res in enumerate(results):
        print(f"  {i+1}. {res['text']} (source: {res['metadata'].get('filename')})")

    if results:
         top_text = results[0]['text']
         if "SpecialKey123" in top_text:
             print("✅ Keyword found via BM25 and ranked #1")
         else:
             print(f"❌ Keyword document found but NOT ranked #1. Top was: {top_text}")

    # 4. Search - Semantic (Vector preferred - mocked)
    print("\n[Test 2] Merging Results")
    results, stats = retriever.search("life", top_k=5)
    found_vector_doc = any('Meaning of life' in r['text'] for r in results)
    print(f"Vector doc found: {found_vector_doc}")
    assert found_vector_doc
    print("✅ Vector result merged")

if __name__ == "__main__":
    test_hybrid_search()

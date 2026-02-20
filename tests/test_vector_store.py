import numpy as np

from app.storage.vector_store import VectorStore


def test_vector_store_add_search_save_load(tmp_path):
    index_path = tmp_path / "index.faiss"
    ids_path = tmp_path / "ids.json"

    dim = 4
    store = VectorStore(index_path=index_path, ids_path=ids_path, dim=dim, normalize=True)

    # Simple, clearly separable vectors (will be normalized internally)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # chunk_a
            [0.0, 1.0, 0.0, 0.0],  # chunk_b
            [0.0, 0.0, 1.0, 0.0],  # chunk_c
        ],
        dtype=np.float32,
    )
    ids = ["chunk_a", "chunk_b", "chunk_c"]

    store.add(embeddings, ids, save=True)
    assert len(store) == 3

    # Query closest to chunk_b
    query = np.array([0.0, 10.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2)

    assert results, "Expected at least one search result"
    assert results[0].chunk_id == "chunk_b"
    assert results[0].score >= results[-1].score  # sorted by FAISS (best first)

    # Reload and ensure it still works
    store2 = VectorStore(index_path=index_path, ids_path=ids_path, dim=dim, normalize=True)
    assert len(store2) == 3

    results2 = store2.search(query, top_k=1)
    assert results2[0].chunk_id == "chunk_b"
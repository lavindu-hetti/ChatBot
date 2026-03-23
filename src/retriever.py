def retrieve_similar_chunks(vectorstore, query: str, k: int = 5, fetch_k: int = 12):
    return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

import json
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb


def load_articles_with_embeddings(folder):
    paths = glob.glob(folder + "/*embedding*.json")
    data = {}
    for path in paths:
        print(path)
        with open(path, "r") as f:
            data.update(json.load(f))
    return data


def embed_text(text, model):
    return model.encode([text])[0]


def search_best_matches(input_text, articles_dict, model, top_k=5):
    input_embedding = embed_text(input_text, model)
    input_embedding = input_embedding.reshape(1, -1)
    results = []
    for url, article in articles_dict.items():
        article_embedding = np.array(article["embedding"])
        article_embedding = article_embedding.reshape(1, -1)
        similarity = cosine_similarity(input_embedding, article_embedding)[0][0]
        results.append(
            {
                "url": url,
                "title": article.get("title", "N/A"),
                "similarity": similarity,
            }
        )
    # Sort by similarity
    sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return sorted_results[:top_k]


def search_best_matches_chroma(
    input_text,
    model,
    chroma_dir="chroma_db",
    collection_name="inferbook_tab_collection1",
    top_k=5,
):
    """
    Search for best matching documents using a ChromaDB collection.

    Assumes documents were stored with:
      - ids = URLs
      - metadata = { 'title': ..., ... }
    and Chroma's default cosine distance metric.
    """
    # Convert query text to embedding using the same model as for file-based search
    query_embedding = embed_text(input_text, model)

    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        raise RuntimeError(
            f"Unable to open ChromaDB collection '{collection_name}' at '{chroma_dir}': {e}"
        )
    res = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["metadatas", "uris", "distances"],
    )

    ids = res.get("ids", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    results = []
    for _id, md, dist in zip(ids, metadatas, distances):
        # Chroma uses distance; for cosine, similarity ~= 1 - distance
        similarity = 1.0 - dist if dist is not None else None
        results.append(
            {
                "url": _id,
                "title": (md or {}).get("title", "N/A"),
                "similarity": similarity,
            }
        )
    # Already ranked; keep same shape as file-based results
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Search best matched articles by embedding similarity."
    )
    parser.add_argument(
        "input_text",
        type=str,
        help="Input text to search for similar articles.",
    )
    parser.add_argument(
        "--backend",
        choices=["file", "chroma"],
        default="file",
        help="Where to search: 'file' (JSON embeddings) or 'chroma' (ChromaDB).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data",
        help="Directory containing embeddings JSON files (used when backend='file').",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="chroma_db",
        help="Directory where the local ChromaDB persistent store lives (backend='chroma').",
    )
    parser.add_argument(
        "--chroma-collection",
        type=str,
        default="inferbook_tab_collection1",
        help="ChromaDB collection name to query (backend='chroma').",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top matches to return.",
    )
    args = parser.parse_args()

    # Use the same model for both backends so embeddings are consistent
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if args.backend == "file":
        articles_dict = load_articles_with_embeddings(args.data)
        best_matches = search_best_matches(
            args.input_text, articles_dict, model, args.top_k
        )
    else:
        best_matches = search_best_matches_chroma(
            args.input_text,
            model,
            chroma_dir=args.chroma_dir,
            collection_name=args.chroma_collection,
            top_k=args.top_k,
        )

    for i, match in enumerate(best_matches, 1):
        print(f"\nRank {i}:")
        similarity = match.get("similarity")
        if similarity is not None:
            print(f"Similarity: {similarity:.4f}")
        else:
            print("Similarity: N/A")
        print(f"Title: {match['title']}")
        print(f"URL: {match['url']}")


if __name__ == "__main__":
    main()
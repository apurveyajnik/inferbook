import json
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import ollama
from config import config, logger
from constants import ModelSources, ModelEmbeddingHF, ModelEmbeddingOllama, \
                    OutputDestination


def load_articles_with_embeddings(folder):
    paths = glob.glob(folder + "/*embedding*.json")
    data = {}
    for path in paths:
        logger.info(path)
        with open(path, "r") as f:
            data.update(json.load(f))
    return data


def load_embedding_model():
     # Use the same model for both backends so embeddings are consistent
    if config['EMBEDDING_MODEL_SOURCE'] == ModelSources.hugging_face.value:
        model_name = ModelEmbeddingHF.ALL_MINI_LM_L6_V2.value
        model = SentenceTransformer(model_name)
    elif config['EMBEDDING_MODEL_SOURCE'] == ModelSources.ollama.value:
        model_name = ModelEmbeddingOllama.ALL_MINI_LM_L6_V2.value
        model = None # model_name added as string in ollama.embed function
        ollama.pull(model_name) # just in case the model is not present locally
    else:
        raise ValueError(f"Unsupported embedding model source: {config['EMBEDDING_MODEL_SOURCE']}")
    
    return model, model_name


def embed_text(text, model, model_name):
    """
    Create an embedding for a single text using the configured embedding backend.

    For HuggingFace, `model` must be a SentenceTransformer instance.
    For Ollama, `model` is ignored and we call the local Ollama embedding model
    defined by ModelEmbeddingOllama.ALL_MINI_LM_L6_V2.
    """
    source = config["EMBEDDING_MODEL_SOURCE"]
    if source == ModelSources.hugging_face.value:
        return model.encode([text])[0]
    elif source == ModelSources.ollama.value:
        return ollama.embed(model=model_name, text=text)
    else:
        raise ValueError(
            f"Unsupported embedding model source: {config['EMBEDDING_MODEL_SOURCE']}"
        )


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
    model_name,
    chroma_dir=config['CHROMA_DIR'],
    collection_name=config['CHROMA_COLLECTION_NAME'],
    top_k=5,
):
    """
    Search for best matching documents using a ChromaDB collection.

    Assumes documents were stored with:
      - ids = URLs
      - metadata = { 'title': ..., ... }
    and Chroma's default cosine distance metric.
    """
    # Convert query text to embedding using the same model backend as for file-based search
    query_embedding = embed_text(input_text, model, model_name)

    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        raise RuntimeError(
            f"Unable to open ChromaDB collection '{collection_name}' at '{chroma_dir}': {e}"
        )
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "uris", "distances"],
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    results = []
    for _id, doc, md, dist in zip(ids, docs, metadatas, distances):
        # Chroma uses distance; for cosine, similarity ~= 1 - distance
        similarity = 1.0 - dist if dist is not None else None
        results.append(
            {
                "url": _id,
                "title": (md or {}).get("title", "N/A"),
                "document": doc,
                "similarity": similarity,
            }
        )
    # Already ranked; keep same shape as file-based results while also
    # exposing the underlying stored document text.
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
        "--top_k",
        type=int,
        default=5,
        help="Number of top matches to return.",
    )
    args = parser.parse_args()

    model, model_name = load_embedding_model()

    if args.backend == OutputDestination.FILE.value:
        articles_dict = load_articles_with_embeddings(args.data)
        best_matches = search_best_matches(
            args.input_text, articles_dict, model, args.top_k
        )
    else:
        best_matches = search_best_matches_chroma(
            args.input_text,
            model,
            model_name,
            chroma_dir=config['CHROMA_DIR'],
            collection_name=config['CHROMA_COLLECTION_NAME'],
            top_k=args.top_k,
        )
    
    dict_matches = {}
    for i, match in enumerate(best_matches, 1):
        logger.info(f"\nRank {i}:")
        similarity = match.get("similarity")
        if similarity is not None:
            logger.info(f"Similarity: {similarity:.4f}")
        else:
            logger.info("Similarity: N/A")
        dict_matches[i] = {
            "title": match['title'],
            "url": match['url'],
            "similarity": similarity,
        }
        logger.info(f"Title: {match['title']}")
        logger.info(f"URL: {match['url']}")

    print(dict_matches)
if __name__ == "__main__":
    main()
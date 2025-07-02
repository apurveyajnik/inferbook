import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_articles_with_embeddings(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def embed_text(text, model):
    return model.encode([text])[0]

def search_best_matches(input_text, articles_dict, model, top_k=5):
    input_embedding = embed_text(input_text, model)
    input_embedding = input_embedding.reshape(1, -1)
    results = []
    for url, article in articles_dict.items():
        article_embedding = np.array(article['embedding'])
        article_embedding = article_embedding.reshape(1, -1)
        similarity = cosine_similarity(input_embedding, article_embedding)[0][0]
        results.append({
            'url': url,
            'title': article.get('title', 'N/A'),
            'similarity': similarity
        })
    # Sort by similarity
    sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return sorted_results[:top_k]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Search best matched articles by embedding similarity.")
    parser.add_argument("input_text", type=str, help="Input text to search for similar articles.")
    parser.add_argument("--data", type=str, default="scraped_data_with_embeddings.json", help="Path to JSON file.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top matches to return.")
    args = parser.parse_args()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    articles_dict = load_articles_with_embeddings(args.data)
    best_matches = search_best_matches(args.input_text, articles_dict, model, args.top_k)
    for i, match in enumerate(best_matches, 1):
        print(f"\nRank {i}:")
        print(f"Similarity: {match['similarity']:.4f}")
        print(f"Title: {match['title']}")
        print(f"URL: {match['url']}")

if __name__ == "__main__":
    main()
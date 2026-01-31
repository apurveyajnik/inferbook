# pscripts/categorize_articles_llm.py

import json
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import os
import logging
from tqdm import tqdm

# Set up logging to file
logging.basicConfig(
    filename="inferbook.log",
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)

def load_articles_with_embeddings(path):
    logging.info(f"Loading articles with embeddings from {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} articles.")
    return data

def cluster_articles(articles_dict, n_clusters=5):
    urls = list(articles_dict.keys())
    n = len(urls)
    embeddings = np.array([articles_dict[url]['embedding'] for url in urls])
    embeddings = embeddings.reshape(n, -1)
    print(embeddings.shape)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clusters = defaultdict(list)
    for url, label in zip(urls, labels):
        clusters[label].append({
            'url': url,
            'title': articles_dict[url].get('title', 'N/A'),
            'text': articles_dict[url].get('text', []),
        })
    for label, articles in clusters.items():
        logging.info(f"Cluster {label}: {len(articles)} articles.")
    return clusters

def get_cluster_text(articles, max_articles=5, max_chars=2000):
    # Concatenate the first N articles' paragraphs for the cluster
    texts = []
    for article in articles[:max_articles]:
        if article.get('text'):
            texts.append(article['text'])
        elif article.get('title'):
            texts.append(article['title'])
    joined = "\n".join(texts)
    logging.info(f"Generated cluster text for {len(articles[:max_articles])} articles, {len(joined[:max_chars])} chars.")
    return joined[:max_chars]  # Truncate to avoid overly long prompts

def generate_category_title_description(cluster_text, hf_model, hf_api_token=None):
    from transformers import pipeline
    import torch
    # Pass token as a top-level argument if provided
    pipeline_kwargs = {}
    if hf_api_token:
        pipeline_kwargs["token"] = hf_api_token
    generator = pipeline(
        "text-generation",
        model=hf_model,
        tokenizer=hf_model,
        device=0 if torch.cuda.is_available() else -1,
        **pipeline_kwargs
    )
    prompt = (
        "Given the following collection of article excerpts, generate a short, descriptive category title and a 1-2 sentence description summarizing the main theme of these articles.\n\n"
        f"Articles:\n{cluster_text}\n\n"
        "Category Title:\n"
        "Category Description:"
    )
    logging.info(f"Generating category title/description with model {hf_model}.")
    # Generate text
    outputs = generator(prompt, max_new_tokens=128, temperature=0.5, do_sample=True)
    text = outputs[0]["generated_text"]
    logging.info(f"Raw LLM output: {text}")

    # Try to split into title and description more flexibly
    import re
    title, description = "Untitled", ""
    match = re.search(r"Category Title:\s*(.*)\nCategory Description:\s*(.*)", text, re.DOTALL)
    if match:
        title = match.group(1).strip()
        description = match.group(2).strip()
    else:
        # fallback: first line as title, second as description
        lines = text.strip().split('\n')
        if lines:
            title = lines[0].replace("Category Title:", "").strip()
        if len(lines) > 1:
            description = lines[1].replace("Category Description:", "").strip()
    logging.info(f"Generated title: {title}, description: {description}")
    return title, description

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cluster articles and generate learned category titles/descriptions using LLM.")
    parser.add_argument("--data", type=str, default="scraped_data_with_embeddings.json", help="Path to JSON file.")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters/categories.")
    parser.add_argument("--max_articles", type=int, default=5, help="Number of articles to use for LLM prompt per cluster.")
    parser.add_argument("--max_chars", type=int, default=2000, help="Max characters to send to LLM per cluster.")
    parser.add_argument("--hf_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model ID for text generation.")
    parser.add_argument("--output", type=str, default="categories_output.json", help="Output file for categories.")
    args = parser.parse_args()

    hf_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

    articles_dict = load_articles_with_embeddings(args.data)
    clusters = cluster_articles(articles_dict, args.n_clusters)

    output_data = []
    for label, articles in tqdm(clusters.items()):
        cluster_text = get_cluster_text(articles, args.max_articles, args.max_chars)
        title, description = generate_category_title_description(cluster_text, args.hf_model, hf_api_token)
        logging.info(f"Cluster {label+1}: {title} ({len(articles)} articles)")
        logging.info(f"Description: {description}")
        print(f"\n=== Category {label+1}: {title} ({len(articles)} articles) ===")
        print(f"Description: {description}")
        for article in articles[:10]:  # Show up to 10 articles per category
            print(f"- {article['title']} ({article['url']})")
        if len(articles) > 10:
            print(f"...and {len(articles)-10} more.")
        # Only keep url and title for each article
        articles_no_embed = [
            {"url": a["url"], "title": a.get("title", "N/A")}
            for a in articles
        ]
        output_data.append({
            "category": int(label) + 1,
            "title": title,
            "description": description,
            "articles": articles_no_embed
        })

    # Save output to JSON file
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"Saved category results to {args.output}")

if __name__ == "__main__":
    main()
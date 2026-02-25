import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import logging
import json
import functools
import os
from tqdm import tqdm
import argparse
import chromadb

# Configure logging
logging.basicConfig(
    filename='inferbook.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('inferbook')

# Decorator to log function calls
def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Entering function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Exiting function: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in function {func.__name__}: {str(e)}")
            raise
    return wrapper


@log_function_call
def scrape_data(url):
    try:
        # Send a GET request to the URL
        logger.info(f"Fetching URL: {url}")
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'lxml')

        # Extract the title of the page
        title = soup.title.string if soup.title else 'No title found'
        logger.info(f"Extracted title: {title}")

        # Extract all paragraph texts
        paragraphs = [p.text for p in soup.find_all('p')]
        logger.info(f"Extracted {len(paragraphs)} paragraphs")

        return {
            'title': title,
            'paragraphs': paragraphs
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        with open('error.log', 'a') as f:
            f.write(f"Error fetching URL {url}: {str(e)}\n")
        return None
    

@log_function_call
def create_embeddings(texts, model):
    logger.info(f"Creating embeddings for {len(texts)} texts")
    return model.encode(texts)


def _results_embed_to_lists(results_embed):
    """
    Convert results_embed dict into parallel lists suitable for ChromaDB.

    Returns (ids, documents, embeddings, metadatas) where:
      - ids: list of URL strings
      - documents: list of strings "title\n\ntext"
      - embeddings: list of embedding vectors (lists of floats)
      - metadatas: per-item metadata dicts
    """
    ids, embeddings, documents, metadatas = [], [], [], []
    for url, item in results_embed.items():
        embedding = item.get("embedding")
        if embedding is None:
            logger.warning(f"Missing embedding for URL {url}; skipping.")
            continue

        title = item.get("title", "")
        text = item.get("text", "")

        ids.append(url)
        embeddings.append(embedding[0])
        # Combine title and text to form a richer document string
        documents.append(f"{title}\n\n{text}".strip())
        metadatas.append({
            "title": title,
            "folder_path": item.get("folder_path", ""),
            "url": url,
        })

    return ids, documents, embeddings, metadatas


@log_function_call
def save_embeddings_to_chroma(results_embed, collection_name, persist_directory="chroma_db"):
    """
    Save a dictionary of embeddings to a local ChromaDB collection.

    results_embed format (per URL key):
    {
        url: {
            'title': str,
            'text': str,
            'folder_path': str,
            'embedding': List[float]
        }
    }
    """
    logger.info(
        f"Saving {len(results_embed)} embeddings to ChromaDB collection='{collection_name}' "
        f"at path='{persist_directory}'"
    )

    ids, documents, embeddings, metadatas = _results_embed_to_lists(results_embed)
    if not ids:
        logger.info("No valid embeddings to save to ChromaDB.")
        return

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=collection_name)
    import pdb; pdb.set_trace()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info(
        f"Successfully saved {len(ids)} embeddings to ChromaDB collection='{collection_name}'."
    )


@log_function_call
def save_embeddings_file_to_chroma(embeddings_json_path, collection_name, persist_directory="chroma_db"):
    """
    Load embeddings from a JSON file on disk and store them into a ChromaDB collection.

    This is useful if embeddings were previously generated and saved to file.
    """
    logger.info(
        f"Loading embeddings from file '{embeddings_json_path}' into "
        f"ChromaDB collection='{collection_name}' at path='{persist_directory}'"
    )
    if not os.path.exists(embeddings_json_path):
        logger.error(f"Embeddings file not found: {embeddings_json_path}")
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_json_path}")

    with open(embeddings_json_path, "r") as f:
        results_embed = json.load(f)
    if not isinstance(results_embed, dict):
        logger.error("Embeddings file must contain a JSON object/dict.")
        raise ValueError("Embeddings file must contain a JSON object/dict.")
    

    save_embeddings_to_chroma(results_embed, collection_name, persist_directory)


def _build_tree_from_dl(dl_element, tree):
    """Recursively build a nested tree of folders and links from a DL element.

    Tree format:
    {
        "folders": { folder_name: { "folders": {...}, "links": [ {title, href} ] } },
        "links": [ {title, href} ]
    }
    """
    if tree.get('folders') is None:
        tree['folders'] = {}
    if tree.get('links') is None:
        tree['links'] = []

    # Collect DTs whose nearest DL parent is the current DL (robust to <p> wrapper)
    all_dt = dl_element.find_all('dt', recursive=True)
    dt_children = []
    for dt in all_dt:
        parent_dl = dt.find_parent('dl')
        if parent_dl == dl_element:
            dt_children.append(dt)
    logger.info(f"DL had total DTs={len(all_dt)}, direct DT children={len(dt_children)}")

    for dt in dt_children:
        # Folder
        h3 = dt.find('h3', recursive=False)
        if h3:
            folder_name = h3.get_text(strip=True)
            sub_dl = dt.find_next_sibling('dl')
            subtree = {"folders": {}, "links": []}
            tree['folders'][folder_name] = subtree
            if sub_dl:
                _build_tree_from_dl(sub_dl, subtree)
            continue
        # Link
        a = dt.find('a', href=True, recursive=False)
        if a:
            title = a.get_text(strip=True)
            href = a['href']
            tree['links'].append({"title": title, "href": href})


def _flatten_tree_with_paths(tree, current_path=None):
    """Return a flat list of {url, title, folder_path} from the nested tree."""
    if current_path is None:
        current_path = []
    entries = []
    folder_path_str = "/".join(current_path) if current_path else ""
    for link in tree.get('links', []):
        entries.append({
            "url": link.get('href', ''),
            "title": link.get('title', ''),
            "folder_path": folder_path_str
        })
    for name, subtree in tree.get('folders', {}).items():
        entries.extend(_flatten_tree_with_paths(subtree, current_path + [name]))
    return entries


@log_function_call
def process_tab_collection(soup, collection_name):
    logger.info(f"Processing {collection_name}")
    # Case-insensitive, whitespace-tolerant match for the folder heading
    def _match_h3_text(text):
        try:
            return text is not None and text.strip().lower() == collection_name.strip().lower()
        except Exception:
            return False
    tab_collection = soup.find('h3', string=_match_h3_text)
    logger.info(f"Found {collection_name} heading: {tab_collection is not None}")
    
    if not tab_collection:
        return {"folders": {}, "links": []}, []

    parent_dt = tab_collection.find_parent('dt')
    # Prefer immediate sibling DL; fallback to next DL in document
    dl_element = None
    if parent_dt is not None:
        dl_element = parent_dt.find_next_sibling('dl')
        if dl_element is None:
            dl_element = parent_dt.find_next('dl')
    logger.info(f"Resolved DL element: {dl_element is not None}")

    root_tree = {"folders": {}, "links": []}
    if dl_element:
        _build_tree_from_dl(dl_element, root_tree)

    # Log counts for debugging
    def _count(tree):
        num_links = len(tree.get('links', []))
        num_folders = len(tree.get('folders', {}))
        for sub in tree.get('folders', {}).values():
            lf, ff = _count(sub)
            num_links += lf
            num_folders += ff
        return num_links, num_folders
    total_links, total_folders = _count(root_tree)
    logger.info(f"Discovered under '{collection_name}': {total_folders} folders, {total_links} links")

    # Attach the root name as a top-level folder for clarity
    full_tree = {"folders": {collection_name: root_tree}, "links": []}
    entries = _flatten_tree_with_paths(root_tree, [collection_name])
    return full_tree, entries


if __name__ == "__main__":
    logger.info("Starting the scraping and embedding process")

    try:
        # CLI arguments
        parser = argparse.ArgumentParser(description="Scrape bookmarked pages and optionally generate embeddings.")
        parser.add_argument(
            "--save",
            choices=["text", "embeddings"],
            default="embeddings",
            help="Choose 'text' to save only text and titles, or 'embeddings' to also compute embeddings."
        )
        parser.add_argument(
            "--bookmarks",
            default="bookmarks_24_04_2025.html",
            help="Path to the bookmarks HTML file."
        )
        parser.add_argument(
            "--output-destination",
            choices=["file", "chroma", "both"],
            default="file",
            help=(
                "Where to save embeddings when in 'embeddings' mode: "
                "'file' (JSON on disk), 'chroma' (ChromaDB only), or 'both'."
            ),
        )
        parser.add_argument(
            "--chroma-dir",
            default="chroma_db",
            help="Directory where the local ChromaDB persistent store will be created/used.",
        )
        parser.add_argument(
            "--chroma-collection-prefix",
            default="inferbook",
            help="Prefix for ChromaDB collection names; actual name will be '<prefix>_<collection_name>'.",
        )
        args = parser.parse_args()

        save_mode = args.save  # 'text' or 'embeddings'

        # Conditionally load the embedding model
        model = None
        if save_mode == "embeddings":
            logger.info("Loading embedding model")
            model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load the bookmarks HTML file
        logger.info("Loading bookmarks file")
        with open("htmls/" + args.bookmarks, 'r') as file:
            bookmarks_html = file.read()

        # Parse the HTML content
        soup = BeautifulSoup(bookmarks_html, 'lxml')

        # Process both tab collections
        collections = ['tab_collection1']
        
        for collection_name in collections:
            text_filename = f"data/scraped_data_text_{collection_name}.json"
            embed_filename = f"data/scraped_data_with_embeddings_{collection_name}.json"
            tree_filename = f"data/bookmarks_tree_{collection_name}.json"
            chroma_collection_name = f"{args.chroma_collection_prefix}_{collection_name}"

            # Build nested tree and flat entries with folder paths
            folder_tree, entries = process_tab_collection(soup, collection_name)

            # Save the tree for inspection
            with open(tree_filename, 'w') as tf:
                json.dump(folder_tree, tf, indent=4)

            # Print a brief summary
            if entries:
                print(f"Found {len(entries)} links under '{collection_name}'.")
            else:
                print(f"No links found for {collection_name}.")

            if save_mode == "text":
                logger.info(f"Starting text-only processing for {collection_name}")
                results_text = {}
                for entry in tqdm(entries, desc=f"Processing {collection_name} (text)"):
                    url = entry['url'].strip()
                    logger.info(f"Processing URL: {url}")
                    data = scrape_data(url)
                    if data:
                        combined_text = " ".join(data['paragraphs'])
                        results_text[url] = {
                            'title': data['title'],
                            'text': combined_text,
                            'folder_path': entry['folder_path']
                        }
                logger.info(f"Saving text results to {text_filename}")
                with open(text_filename, 'w') as f:
                    json.dump(results_text, f, indent=4)
                print(f"Completed {collection_name}. Text results saved to '{text_filename}'.")
                continue

            # Embeddings mode
            if os.path.exists(embed_filename):
                logger.info(f"Embeddings JSON already exists for {collection_name}: {embed_filename}")
                print(f"Embeddings already exist for '{collection_name}' in '{embed_filename}'.")

                # If the user wants embeddings in Chroma, load them from the existing file.
                if args.output_destination in ("chroma", "both"):
                    logger.info(
                        f"Saving existing embeddings for {collection_name} from file to ChromaDB collection "
                        f"'{chroma_collection_name}' at '{args.chroma_dir}'."
                    )
                    save_embeddings_file_to_chroma(
                        embed_filename,
                        chroma_collection_name,
                        args.chroma_dir,
                    )
                    print(
                        f"Existing embeddings for '{collection_name}' loaded into "
                        f"ChromaDB collection '{chroma_collection_name}'."
                    )
                # Skip recomputing embeddings if they already exist on disk.
                continue

            # Ensure text data exists; load or create it
            results_text = None
            if os.path.exists(text_filename):
                logger.info(f"Loading existing text data from {text_filename}")
                with open(text_filename, 'r') as f:
                    results_text = json.load(f)
            else:
                logger.info(f"Text data not found for {collection_name}. Creating text JSON first: {text_filename}")
                results_text = {}
                for entry in tqdm(entries, desc=f"Processing {collection_name} (prepare text)"):
                    url = entry['url'].strip()
                    logger.info(f"Processing URL: {url}")
                    data = scrape_data(url)
                    if data:
                        combined_text = " ".join(data['paragraphs'])
                        results_text[url] = {
                            'title': data['title'],
                            'text': combined_text,
                            'folder_path': entry['folder_path']
                        }
                logger.info(f"Saving generated text results to {text_filename}")
                with open(text_filename, 'w') as f:
                    json.dump(results_text, f, indent=4)
                print(f"Prepared text data for {collection_name} at '{text_filename}'.")

            # Create embeddings based on text data
            logger.info(f"Starting embeddings generation for {collection_name}")
            
            results_embed = {}
            for url, item in tqdm(results_text.items(), desc=f"Processing {collection_name} (embeddings)"):
                combined_text = item.get('text', '')
                embedding = create_embeddings([combined_text], model)
                results_embed[url] = {
                    'title': item.get('title', ''),
                    'text': combined_text,
                    'folder_path': item.get('folder_path', ''),
                    'embedding': embedding.tolist()
                }

            # Save embeddings to requested destinations
            if args.output_destination in ("file", "both"):
                logger.info(f"Saving embeddings results to {embed_filename}")
                with open(embed_filename, 'w') as f:
                    json.dump(results_embed, f, indent=4)
                print(f"Completed {collection_name}. Embeddings saved to '{embed_filename}'.")

            if args.output_destination in ("chroma", "both"):
                logger.info(
                    f"Saving embeddings for {collection_name} to ChromaDB collection "
                    f"'{chroma_collection_name}' at '{args.chroma_dir}'."
                )
                
                save_embeddings_to_chroma(
                    results_embed,
                    chroma_collection_name,
                    args.chroma_dir,
                )
                print(
                    f"Completed {collection_name}. Embeddings saved to ChromaDB collection "
                    f"'{chroma_collection_name}'."
                )

        logger.info("Scraping and embedding completed successfully")
        print("Processing completed for all collections.")
        
    except Exception as e:
        logger.error(f"An error occurred in the main process: {str(e)}")
        raise
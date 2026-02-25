# InferBook

This folder contains a module for scraping bookmarked pages, generating embeddings, and querying them.

Below are instructions for:
- `src/scraper.py`: scrape bookmarks, build text JSON, and optionally create/store embeddings.
- `src/infer.py`: query the generated embeddings (from JSON files or ChromaDB).

## inferbook/src/scraper.py

Scrapes bookmarked pages from an exported bookmarks HTML file and optionally generates sentence embeddings.

### Requirements
- Python 3.10+
- Install dependencies (example):
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, minimally install:
```bash
pip install requests beautifulsoup4 sentence-transformers tqdm
```

### Usage
```bash
python src/scraper.py \
  [--save {text,embeddings}] \
  [--bookmarks FILENAME] \
  [--output-destination {file,chroma,both}] \
  [--chroma-dir DIR] \
  [--chroma-folder-prefix PREFIX]
```

- **--save**: Choose output mode. Defaults to `embeddings`.
  - `text`: Save only title and extracted text.
  - `embeddings`: Save title, text, and embedding vector, and optionally write to ChromaDB.
- **--bookmarks**: Filename of the bookmarks HTML file (the script reads from `htmls/<FILENAME>`).  
  Default: `bookmarks_%d_%m_%YYYY.html`.
- **--output-destination**: Where to save embeddings *when* `--save embeddings` is used.  
  Choices: `file`, `chroma`, `both`. Default: `file`.
  - `file`: Save embeddings JSON to disk only.
  - `chroma`: Store embeddings only in a local ChromaDB folder.
  - `both`: Save to JSON and also store in ChromaDB.
- **--chroma-dir**: Directory for the local ChromaDB persistent store.  
  Default: `chroma_db`.
- **--chroma-folder-prefix**: Prefix for ChromaDB folder names.  
  Actual folder used is `<prefix>_<folder_name>` (e.g., `inferbook_tab_sample_folder1`).  
  Default: `inferbook`.

### Examples
- Save only text:
```bash
python src/scraper.py --save text --bookmarks bookmarks_%d_%m_%YYYY.html
```

- Save text and embeddings to JSON files (default behavior):
```bash
python src/scraper.py --save embeddings --bookmarks bookmarks_%d_%m_%YYYY.html
```

- Save embeddings **only** into ChromaDB:
```bash
python src/scraper.py \
  --save embeddings \
  --bookmarks bookmarks_%d_%m_%YYYY.html \
  --output-destination chroma
```

- Save embeddings to **both** JSON and ChromaDB with a custom DB directory and prefix:
```bash
python src/scraper.py \
  --save embeddings \
  --bookmarks bookmarks_%d_%m_%YYYY.html \
  --output-destination both \
  --chroma-dir my_chroma_db \
  --chroma-folder-prefix my_inferbook
```

### Outputs
For each tab folder processed (e.g., `tab_sample_folder1`, `tab_sample_folder2`):
- When `--save text`:
  - `data/scraped_data_text_tab_sample_folder1.json`
  - `data/scraped_data_text_tab_sample_folder2.json`
- When `--save embeddings`:
  - `data/scraped_data_with_embeddings_tab_sample_folder1.json`
  - `data/scraped_data_with_embeddings_tab_sample_folder2.json`

Each JSON maps URL to an object containing:
- `title`: Page title
- `text`: Combined page paragraph text
- `embedding` (only in embeddings mode): List[float]

### Logs
- Operational logs: `inferbook.log`
- Network and Scraping errors: `error.log` 


## inferbook/src/infer.py

Search for best-matching articles using the generated embeddings, either from JSON files on disk or from a ChromaDB folder.

### Usage
```bash
python src/infer.py \
  "your query text here" \
  [--backend {file,chroma}] \
  [--data DATA_DIR] \
  [--chroma-dir DIR] \
  [--chroma-folder NAME] \
  [--top_k N]
```

- **input_text (positional)**: The query text you want to search with.
- **--backend**: Where to search for embeddings.  
  Choices: `file`, `chroma`. Default: `file`.
  - `file`: Reads `*embedding*.json` files from `--data` directory.
  - `chroma`: Queries a ChromaDB folder in `--chroma-dir`.
- **--data**: Directory containing embeddings JSON files (used when `--backend file`).  
  Default: `./data`.
- **--chroma-dir**: Directory where the local ChromaDB persistent store lives (used when `--backend chroma`).  
  Default: `chroma_db`.
- **--chroma-folder**: ChromaDB folder name to query (used when `--backend chroma`).  
  Default: `inferbook_tab_folder1`.
- **--top_k**: Number of top matches to return.  
  Default: `5`.

### Examples
- Search using JSON embeddings on disk:
```bash
python src/infer.py "how to creat a bookmark search and chat" \
  --backend file \
  --data ./data \
  --top_k 5
```

- Search using a ChromaDB folder:
```bash
python src/infer.py "how to create a bookmark search and chat" \
  --backend chroma \
  --chroma-dir chroma_db \
  --chroma-folder inferbook_tab_folder1 \
  --top_k 5
```

The script prints ranked matches with similarity scores, titles, and URLs.

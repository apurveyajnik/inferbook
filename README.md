# InferBook

This folder contains utility scripts. Below are instructions to use the InferBook scraper.

## inferbook/scraper.py

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
python pscripts/inferbook/scraper.py [--save {text,embeddings}] [--bookmarks PATH]
```

- **--save**: Choose output mode. Defaults to `embeddings`.
  - `text`: Save only title and extracted text.
  - `embeddings`: Save title, text, and embedding vector.
- **--bookmarks**: Path to the bookmarks HTML file. Defaults to `bookmarks_24_04_2025.html` in the current directory.

### Examples
- Save only text:
```bash
python pscripts/inferbook/scraper.py --save text --bookmarks pscripts/inferbook/bookmarks_24_04_2025.html
```

- Save text and embeddings (default):
```bash
python pscripts/inferbook/scraper.py --save embeddings --bookmarks pscripts/inferbook/bookmarks_24_04_2025.html
```

### Outputs
For each tab collection processed (e.g., `tab_collection1`, `tab_collection2`):
- When `--save text`:
  - `scraped_data_text_tab_collection1.json`
  - `scraped_data_text_tab_collection2.json`
- When `--save embeddings`:
  - `scraped_data_with_embeddings_tab_collection1.json`
  - `scraped_data_with_embeddings_tab_collection2.json`

Each JSON maps URL to an object containing:
- `title`: Page title
- `text`: Combined page paragraph text
- `embedding` (only in embeddings mode): List[float]

### Logs
- Operational logs: `inferbook.log`
- Network errors: `error.log` 
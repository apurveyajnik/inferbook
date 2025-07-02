import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import logging
import json
import functools
from tqdm import tqdm

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
        soup = BeautifulSoup(response.text, 'html.parser')

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


if __name__ == "__main__":
    logger.info("Starting the scraping and embedding process")

    try:
        # Load the embedding model
        logger.info("Loading embedding model")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load the bookmarks HTML file
        logger.info("Loading bookmarks file")
        with open('bookmarks_24_04_2025.html', 'r') as file:
            bookmarks_html = file.read()

        # Parse the HTML content
        soup = BeautifulSoup(bookmarks_html, 'html.parser')

        # Find the "tab_collection1" heading
        tab_collection1 = soup.find('h3', string='tab_collection1')
        logger.info(f"Found tab_collection1 heading: {tab_collection1 is not None}")
        
        urls = []
        if tab_collection1:
            
            if tab_collection1:
                # Get the next DL element after the parent DT
                dl_element = tab_collection1.find_next_sibling('dl')
                logger.info(f"Found DL element: {dl_element is not None}")
                
                if dl_element:
                    # Get only the direct DT elements that contain links
                    dt_elements = dl_element.find_all('dt')
                    logger.info(f"Found {len(dt_elements)} DT elements")
                    
                    for dt in dt_elements:
                        link = dt.find('a', href=True)
                        if link:
                            urls.append(link['href'])
                            logger.info(f"Found link: {link['href']}")

        # Dictionary to store results
        results = {}

        for url in tqdm(urls):
            url = url.strip()
            logger.info(f"Processing URL: {url}")
            data = scrape_data(url)
            if data:
                # Combine all paragraphs into a single text for embedding
                combined_text = " ".join(data['paragraphs'])
                embedding = create_embeddings([combined_text], model)

                # Store the results
                results[url] = {
                    'title': data['title'],
                    'embedding': embedding.tolist(),
                    'text': combined_text
                }

        # Save results to a JSON file
        logger.info("Saving results to JSON file")
        with open('scraped_data_with_embeddings.json', 'w') as f:
            json.dump(results, f, indent=4)

        logger.info("Scraping and embedding completed successfully")
        print("Scraping and embedding completed. Results saved to 'scraped_data_with_embeddings.json'.")
        
    except Exception as e:
        logger.error(f"An error occurred in the main process: {str(e)}")
        raise
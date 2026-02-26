import os
import logging
from dotenv import load_dotenv

load_dotenv()


config = {
    'EMBEDDING_MODEL_SOURCE': os.environ['EMBEDDING_MODEL_SOURCE'],
    'CHAT_MODEL_SOURCE': os.environ['CHAT_MODEL_SOURCE'],
    'OPENAI_PROJECT_KEY': os.environ['OPENAI_PROJECT_KEY'],
    'HUGGINGFACE_API_TOK': os.environ['HUGGINGFACE_API_TOK'],
    'OLLAMA_BASE_URL': os.environ['OLLAMA_BASE_URL'],
    'CHROMA_DIR': os.environ['CHROMA_DIR'],
    'CHROMA_COLLECTION_NAME': os.environ['CHROMA_COLLECTION_NAME']
}

# Configure logging
logging.basicConfig(
    filename='inferbook.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('inferbook')
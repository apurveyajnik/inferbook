from enum import Enum

class OutputDestination(Enum):
    FILE = "file"
    CHROMA = "chroma"
    BOTH = "both"


class ChromaDBFolderPrefix(Enum):
    INFERBOOK = "inferbook"


class ModelSources(Enum):
    hugging_face = "hugging_face"
    ollama = "ollama"
    openai = "openai"
    anthropic = "anthropic"
    mistral = "mistral"
    llama = "llama"


class ModelEmbeddingHF(Enum):
    ALL_MINI_LM_L6_V2 = "all-MiniLM-L6-v2"


class ModelEmbeddingOllama(Enum):
    ALL_MINI_LM_L6_V2 = "all-minilm:l6-v2"
    ALL_MINI_LM_M22 = "all-minilm:22m"


class ModelChatOllama(Enum):
    QWEN_3 = "qwen3:latest"
    QWEN_3_0_6B = "qwen3:0.6b"
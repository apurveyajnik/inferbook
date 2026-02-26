from config import config, logger
from typing import List, Dict, Any
import ollama

from constants import ModelSources, ModelChatOllama
from infer import load_embedding_model, search_best_matches_chroma



def _build_context_from_docs(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "No relevant documents were retrieved from the knowledge base."

    parts = []
    for i, d in enumerate(docs, 1):
        title = d.get("title") or "N/A"
        url = d.get("url") or "N/A"
        similarity = d.get("similarity")
        sim_str = f"{similarity:.4f}" if similarity is not None else "N/A"
        doc_text = d.get("document") or ""
        snippet = doc_text[:2000]
        parts.append(
            f"[Source {i}] Title: {title}\nURL: {url}\nSimilarity: {sim_str}\nContent:\n{snippet}"
        )
    return "\n\n".join(parts)


def run_chat(
    chroma_dir: str = config['CHROMA_DIR'],
    collection_name: str = config['CHROMA_COLLECTION_NAME'],
    top_k: int = 5,
) -> None:
    """
    Terminal chat:
      - reads runtime config from environment via src/config.py
      - uses infer.load_embedding_model + infer.search_best_matches_chroma
      - calls local Ollama QWEN_3_0_6B with Chroma docs as context.
    """
    base = config['OLLAMA_BASE_URL']
    config["OLLAMA_HOST"] = base

    logger.info(
        f"Starting chat session with OLLAMA_BASE_URL={base}, "
        f"EMBEDDING_MODEL_SOURCE={config['EMBEDDING_MODEL_SOURCE']}, "
        f"CHAT_MODEL_SOURCE={config['CHAT_MODEL_SOURCE']}, "
        f"CHROMA_DIR={config['CHROMA_DIR']}, "
        f"CHROMA_COLLECTION_NAME={config['CHROMA_COLLECTION_NAME']}, top_k={top_k}"
    )

    if config['CHAT_MODEL_SOURCE'] != ModelSources.ollama.value:
        logger.error(
            f"Unsupported CHAT_MODEL_SOURCE={config['CHAT_MODEL_SOURCE']} for chat.py; "
            "only 'ollama' is supported."
        )
        raise RuntimeError("chat.py currently only supports Ollama as the chat backend.")

    embed_model, embed_model_name = load_embedding_model()
    logger.info(
        f"Loaded embedding model. EMBEDDING_MODEL_SOURCE={config['EMBEDDING_MODEL_SOURCE']}, "
        f"model_name={embed_model_name}"
    )
    chat_model = ModelChatOllama.QWEN_3_0_6B.value

    system_prompt = (
        "You are a helpful assistant. Use the provided 'Context from knowledge base' "
        "to answer the user's question. If the context is insufficient or unrelated, "
        "say so and answer as best you can, clearly marking any speculation."
    )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print("Inferbook Chat (Qwen 3 0.6B via Ollama, with Chroma context)")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            logger.info("Chat session terminated by user (EOF/KeyboardInterrupt).")
            break

        if not user_input:
            logger.debug("Empty user input received; prompting again.")
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            logger.info("User requested exit from chat via command.")
            break

        # Retrieve closest docs from Chroma using infer.search_best_matches_chroma
        try:
            logger.info(
                f"Querying ChromaDB for user input (len={len(user_input)}): {user_input[:200]}..."
            )
            docs = search_best_matches_chroma(
                user_input,
                embed_model,
                embed_model_name,
                chroma_dir=chroma_dir,
                collection_name=collection_name,
                top_k=top_k,
            )
        except Exception as e:
            print(f"[Error] Failed to query ChromaDB: {e}")
            logger.error(f"Failed to query ChromaDB: {e}")
            docs = []
        else:
            logger.info(f"Retrieved {len(docs)} documents from ChromaDB.")
            if docs:
                sample_urls = [d.get('url') for d in docs[:3]]
                logger.debug(f"Top ChromaDB result URLs: {sample_urls}")

        context = _build_context_from_docs(docs)

        augmented_user = (
            "Context from knowledge base:\n"
            f"{context}\n\n"
            "User question:\n"
            f"{user_input}"
        )
        messages.append({"role": "user", "content": augmented_user})

        try:
            logger.info(
                f"Calling Ollama chat with model={chat_model} and "
                f"{len(messages)} total messages in history."
            )
            response = ollama.chat(
                model=chat_model,
                messages=messages,
            )
        except Exception as e:
            print(f"[Error] Failed to call Ollama chat: {e}")
            logger.error(f"Failed to call Ollama chat: {e}")
            messages.pop()
            continue

        assistant_content = response.get("message", {}).get("content", "").strip()
        if not assistant_content:
            assistant_content = "[No response content returned from model.]"
            logger.warning("Ollama chat returned empty content.")
        else:
            logger.info(
                f"Ollama chat response length={len(assistant_content)} characters."
            )

        print(f"\nAssistant:\n{assistant_content}\n")

        messages.append({"role": "assistant", "content": assistant_content})

        if docs:
            print("Sources:")
            for i, d in enumerate(docs, 1):
                title = d.get("title") or "N/A"
                url = d.get("url") or "N/A"
                print(f"  [{i}] {title} - {url}")
            logger.debug(
                "Displayed Chroma sources to user: "
                f"{[(d.get('title'), d.get('url')) for d in docs]}"
            )
            print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Inferbook chat using local Ollama (QWEN_3_0_6B) and ChromaDB for context."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest documents from Chroma to inject into context.",
    )
    args = parser.parse_args()

    run_chat(
        chroma_dir=config['CHROMA_DIR'],
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()


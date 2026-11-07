"""Interactive chatbot entry point."""

from resources.document_cache import ensure_document_current
from tasks.rag import create_chatbot
from tasks.vector_store import get_vector_store


def run_chat(chatbot) -> None:
    """Read and answer questions until the user exits."""
    print("Ask questions about your travel documents.")
    print("Commands: /clear resets memory; /exit closes the chatbot.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        command = question.lower()
        if command in {"/exit", "exit", "quit"}:
            print("Goodbye!")
            break
        if command in {"/clear", "clear"}:
            chatbot.clear_history()
            print("Bot: Conversation memory cleared.")
            continue

        try:
            answer = chatbot.ask(question)
            print(f"Bot: {answer}")
        except Exception as error:
            print(f"Bot error: {error}")


def main():
    """Prepare application resources and start the interactive chatbot."""
    print("Hello, Welcome!")
    document_changed, metadata = ensure_document_current()
    vector_store = get_vector_store(document_changed, metadata)
    chatbot = create_chatbot(vector_store)
    run_chat(chatbot)


if __name__ == "__main__":
    main()

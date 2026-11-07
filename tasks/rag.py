"""Configure conversational retrieval and question answering."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from resources.config import (
    MAX_HISTORY_MESSAGES,
    RETRIEVER_DOCUMENT_COUNT,
)
from resources.model_api import create_chat_model


CONTEXTUALIZE_SYSTEM_PROMPT = """Rewrite the user's latest question as a
standalone search query using the conversation history. Preserve the user's
meaning. Return only the rewritten question, without answering it."""

ANSWER_SYSTEM_PROMPT = """You answer questions using the retrieved excerpts
from the travel documents. Focus on substantive content and
synthesize information across excerpts. Do not discuss file metadata or
document formatting. If the excerpts do not contain enough information, say
that you don't know.

Retrieved excerpts:
{context}"""


def create_retriever(vector_store, k: int = RETRIEVER_DOCUMENT_COUNT):
    """Create a retriever that returns the top k matching documents."""
    return vector_store.as_retriever(search_kwargs={"k": k})


def format_documents(documents) -> str:
    """Render retrieved page content without exposing PDF metadata."""
    excerpts = []
    for number, document in enumerate(documents, start=1):
        page = document.metadata.get("page_label") or document.metadata.get("page")
        page_note = f", page {page}" if page is not None else ""
        excerpts.append(
            f"[Excerpt {number}{page_note}]\n{document.page_content}"
        )
    return "\n\n---\n\n".join(excerpts)


class RagChatbot:
    """A conversational chatbot grounded in the indexed document."""

    def __init__(self, vector_store):
        self.chat_history: list[BaseMessage] = []
        self.retriever = create_retriever(vector_store)
        self.model = create_chat_model()

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        self.contextualize_chain = (
            contextualize_prompt | self.model | StrOutputParser()
        )

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ANSWER_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        self.answer_chain = answer_prompt | self.model | StrOutputParser()

    def ask(self, question: str) -> str:
        """Answer one question and add the exchange to conversation memory."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        if self.chat_history:
            search_query = self.contextualize_chain.invoke(
                {
                    "chat_history": self.chat_history,
                    "question": question,
                }
            )
        else:
            search_query = question

        documents = self.retriever.invoke(search_query)
        answer = self.answer_chain.invoke(
            {
                "chat_history": self.chat_history,
                "question": question,
                "context": format_documents(documents),
            }
        )

        self.chat_history.extend(
            [HumanMessage(content=question), AIMessage(content=answer)]
        )
        self.chat_history = self.chat_history[-MAX_HISTORY_MESSAGES:]
        return answer

    def clear_history(self) -> None:
        """Forget all messages from the current chat session."""
        self.chat_history.clear()


def create_chatbot(vector_store) -> RagChatbot:
    """Create a conversational RAG chatbot."""
    return RagChatbot(vector_store)

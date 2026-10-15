from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document


# Load environment variables
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)


# Configure the LLM model hosted through OpenRouter.
MODEL = "openai/gpt-4o-mini"

# Path to the persistent ChromaDB vector database created during ingestion.
DB_NAME = str(Path(__file__).parent.parent / "vector_db")


# Number of relevant document chunks to retrieve for each question.
RETRIEVAL_K = 8


# System prompt that defines the assistant's role and response behavior.
# The {context} placeholder will be replaced with retrieved document content.
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""


# Initialize the same embedding model used during ingestion.
# It converts the user's question into a vector so ChromaDB can find
# semantically similar document chunks.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to the existing persistent ChromaDB database.
# The embedding function is used to embed incoming search queries.
vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings
)

# Create a retriever that returns the 8 most relevant chunks by default.
# These chunks will provide context for answering the user's question.
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


# Initialize the chat model via OpenRouter.
# The API key authenticates requests, and temperature=0 encourages
# more consistent responses.
llm = ChatOpenAI(
    model=MODEL,
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)


# Retrieve relevant document chunks from ChromaDB.
def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant documents from the vector database
    """
    return retriever.invoke(question)


#Combine previous user questions with the current question to improve
#retrieval.
def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine prior conversation history with the current question
    to improve retrieval.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


#Run the RAG (Retrieval-Augmented Generation) pipeline.
#1. Combine the current question with previous user questions.
#2. Retrieve relevant chunks from ChromaDB.
#3. Combine retrieved chunks into a context string.
#4. Insert that context into the system prompt.
#5. Send the prompt, conversation history, and current question
#to the LLM.
#6. Return the generated answer and retrieved documents.

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    RAG pipeline: retrieve context + generate grounded answer
    """
    # Include previous user questions to improve retrieval for follow-ups.
    combined = combined_question(question, history)
    #Retrieve relevant chunks from ChromaDB.
    docs = fetch_context(combined)
    #Combine retrieved chunks into a context string.
    context = "\n\n".join(doc.page_content for doc in docs)
    #Insert that context into the system prompt.
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    
    # Return both the answer and retrieved documents.
    # The documents can be displayed in the UI as supporting context.
    response = llm.invoke(messages)
    return response.content, docs

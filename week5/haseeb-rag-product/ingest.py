import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from the .env file.
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# Define the directory where ChromaDB will persist the vector database.
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# Define the directory containing the source Markdown documents.
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# Configure the embedding model.
# Embeddings convert text into numerical vectors that capture semantic meaning.
# These vectors allow us to retrieve documents based on meaning, not just keywords.

# Option 1: OpenAI embeddings (requires an OpenAI API key and applicable API access).
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Option 2: Local Hugging Face embeddings (runs locally without an OpenAI API key).
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Load the source documents
def fetch_documents():
    # Find the immediate subfolders inside the knowledge base.
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        # Use the folder name as the document category.
        doc_type = os.path.basename(folder)

        # Load all Markdown files inside this folder and its subfolders.
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )

        # Each loaded document contains page_content and metadata.
        folder_docs = loader.load()
        
        # Store the category in metadata so we can identify the
        # document type when retrieving relevant context later.
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


#Split documents into smaller, overlapping text chunks.
def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    return chunks

#Convert chunks into vectors using the embedding model and store them in ChromaDB
def create_embeddings(chunks):
    # Delete the existing collection if the database directory exists.
    # This prevents old and new ingestion results from being mixed.
    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings
        ).delete_collection()

    # Create a new ChromaDB instance and store the embedded chunks.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )

    # Get the total count of stored vectors (chunks).
    collection = vectorstore._collection
    count = collection.count()

    # Retrieve one stored embedding to check its vector dimensions.
    sample_embedding = collection.get(
        limit=1,
        include=["embeddings"]
    )["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors stored")
    print(f"Embedding dimensions: {dimensions}")

    return vectorstore


if __name__ == "__main__":
    # Step 1: Load the source documents.
    documents = fetch_documents()
    # Step 2: Split the documents into chunks.
    chunks = create_chunks(documents)
    # Step 3: Create embeddings for the chunks and store them in the vector database.
    create_embeddings(chunks)
    print("ingestion complete")

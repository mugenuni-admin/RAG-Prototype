import os
import sys
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables (API Key)
load_dotenv()

DATA_DIR = "./data"
CHROMA_DB_DIR = "./chroma_db"

def load_documents():
    documents = []
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found. Creating it...")
        os.makedirs(DATA_DIR)
        return documents

    # Get all files in the data directory recursively
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        files = [sys.argv[1]]
        print(f"Only processing specific file: {sys.argv[1]}")
    else:
        files = glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True)
    
    for file_path in files:
        print(f"Loading {file_path}...")
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            else:
                print(f"Skipping unsupported file type: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return documents

def main():
    print("Starting data ingestion process...")
    
    # 1. Load Documents
    docs = load_documents()
    if not docs:
        print(f"No documents found in {DATA_DIR}. Please add some PDFs, TXTs, or DOCXs and run again.")
        return
        
    print(f"Loaded {len(docs)} document pages/sections.")
    
    # 2. Split into chunks (paragraphs)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split documents into {len(splits)} chunks.")
    
    # 3. Create Embeddings and Store in Vector Database (Pinecone)
    if "PINECONE_API_KEY" not in os.environ:
        print("Error: PINECONE_API_KEY not found in environment variables.")
        return

    print("Generating embeddings and uploading to Pinecone...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    index_name = "mugenuni-data-room"
    
    vectorstore = PineconeVectorStore.from_documents(
        documents=splits, 
        embedding=embeddings, 
        index_name=index_name
    )
    
    print("Success! Data Room uploaded to Pinecone index: " + index_name)

if __name__ == "__main__":
    main()

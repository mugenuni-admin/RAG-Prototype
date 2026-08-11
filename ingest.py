import os
import sys
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import base64

# Load environment variables (API Key)
load_dotenv()

import sys

# Allow overriding the data directory via command-line argument
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "./data"
CHROMA_DB_DIR = "./chroma_db"

import time
def get_image_description(file_path):
    if "plate of uni" in file_path.lower():
        return "a beautifully presented plate of fresh, high-quality uni (sea urchin roe) showcasing its characteristic firm texture and vibrant mango-orange color. These delicate, tongue-shaped lobes are arranged neatly, ready to be served as a premium culinary delicacy."

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in detail. Make sure to note its contents, colors, objects, and any text present. Be very descriptive as this will be used for a search engine."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_string}"}
        ]
    )
    
    # Add manual retry logic for 503 errors
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = llm.invoke([message])
            content = response.content
            if isinstance(content, list):
                text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and "text" in part]
                return " ".join(text_parts) if text_parts else str(content)
            return str(content)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Gemini API error (attempt {attempt+1}): {e}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise e

def load_documents():
    documents = []
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found. Creating it...")
        os.makedirs(DATA_DIR)
        return documents

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        if os.path.isfile(target_path):
            files = [target_path]
            print(f"Only processing specific file: {target_path}")
        elif os.path.isdir(target_path):
            files = glob.glob(os.path.join(target_path, "**", "*"), recursive=True)
            print(f"Only processing specific directory: {target_path}")
        else:
            print(f"Error: Path '{target_path}' not found.")
            return []
    else:
        files = glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True)
        print("Processing all files in data directory...")
    
    for file_path in files:
        print(f"Loading {file_path}...")
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            elif file_path.lower().endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file_path.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"Generating image description for {file_path}...")
                description = get_image_description(file_path)
                doc = Document(page_content=f"[IMAGE CAPTION]: {description}", metadata={"source": file_path, "type": "image"})
                documents.append(doc)
            else:
                print(f"Skipping unsupported file type: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise e
            raise e
            
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
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            vectorstore = PineconeVectorStore.from_documents(
                documents=splits, 
                embedding=embeddings, 
                index_name=index_name
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Pinecone/Embedding error (attempt {attempt+1}): {e}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise e
    
    print(f"Success! Data Room uploaded to Pinecone index: {index_name}")

if __name__ == "__main__":
    main()

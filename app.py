import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
import db
from pdf_utils import generate_watermarked_pdf

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="My Data Room", page_icon="🔒", layout="centered")

# Hide the sidebar if not authenticated
if not st.session_state.get("authentication_status"):
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

with open('auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

try:
    authenticator.login()
except Exception as e:
    st.error(f"Login Error: {e}")

if st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
elif st.session_state["authentication_status"]:
    # --- Authenticated App ---
    with st.sidebar:
        st.write(f'Welcome *{st.session_state["name"]}*')
        authenticator.logout(location='sidebar')
        st.divider()

    st.title("🔒 My Secure Data Room")
    st.write("Ask questions about your documents. Only the answers are shown.")

    @st.cache_resource
    def setup_rag_chain():
        if "PINECONE_API_KEY" not in os.environ:
            st.error("PINECONE_API_KEY not found in environment variables.")
            return None
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        index_name = "mugenuni-data-room"
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 30}) 
        
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
        
        system_prompt = (
            "You are a helpful assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        llm_chain = prompt | llm | StrOutputParser()
        
        return {"retriever": retriever, "llm_chain": llm_chain}

    # --- UI and Logic ---
    with st.sidebar:
        st.header("📁 Add More Sources")
        uploaded_files = st.file_uploader("Upload PDF, TXT, or DOCX", accept_multiple_files=True, type=['pdf', 'txt', 'docx'])
        if uploaded_files:
            if st.button("Process & Add to Data Room"):
                with st.spinner("Saving and Processing..."):
                    if not os.path.exists("data"):
                        os.makedirs("data")
                    for f in uploaded_files:
                        with open(os.path.join("data", f.name), "wb") as f_out:
                            f_out.write(f.getbuffer())
                    
                    import subprocess
                    subprocess.run(["python", "ingest.py"], cwd=os.path.dirname(__file__))
                    st.cache_resource.clear()
                    
                    st.success("Successfully added to the data room!")
                    st.rerun()

        st.divider()
        st.subheader("Import from Google Drive")
        gdrive_url = st.text_input("Paste Google Doc or Folder Link")
        if gdrive_url:
            if st.button("Import to Data Room"):
                import re
                doc_match = re.search(r"/d/([a-zA-Z0-9-_]+)", gdrive_url)
                folder_match = re.search(r"/folders/([a-zA-Z0-9-_]+)", gdrive_url)
                
                if not doc_match and not folder_match:
                    st.error("Invalid Google Drive URL.")
                else:
                    with st.spinner("Authenticating & loading from Google Drive..."):
                        try:
                            from langchain_google_community import GoogleDriveLoader
                            from langchain_pinecone import PineconeVectorStore
                            from langchain_google_genai import GoogleGenerativeAIEmbeddings
                            from langchain_text_splitters import RecursiveCharacterTextSplitter
                            from google.oauth2.credentials import Credentials
                            from google_auth_oauthlib.flow import InstalledAppFlow
                            from google.auth.transport.requests import Request
                            
                            creds = None
                            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
                            if os.path.exists("token.json"):
                                creds = Credentials.from_authorized_user_file("token.json", SCOPES)
                            if not creds or not creds.valid:
                                if creds and creds.expired and creds.refresh_token:
                                    creds.refresh(Request())
                                else:
                                    flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                                    creds = flow.run_local_server(port=0)
                                with open("token.json", "w") as token:
                                    token.write(creds.to_json())

                            docs = []
                            if folder_match:
                                loader = GoogleDriveLoader(folder_id=folder_match.group(1), credentials=creds, recursive=False)
                                docs = loader.load()
                            elif doc_match:
                                loader = GoogleDriveLoader(document_ids=[doc_match.group(1)], credentials=creds)
                                docs = [loader._load_document_from_id(doc_match.group(1))]
                            
                            if docs:
                                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
                                splits = text_splitter.split_documents(docs)
                                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
                                
                                import time
                                index_name = "mugenuni-data-room"
                                vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
                                batch_size = 20
                                progress_text = st.empty()
                                
                                for i in range(0, len(splits), batch_size):
                                    batch = splits[i:i + batch_size]
                                    progress_text.write(f"⏳ Embedding chunks {i+1} to {min(i+batch_size, len(splits))} of {len(splits)}...")
                                    max_retries = 4
                                    for attempt in range(max_retries):
                                        try:
                                            vectorstore.add_documents(documents=batch)
                                            break
                                        except Exception as e:
                                            if '429' in str(e) and attempt < max_retries - 1:
                                                time.sleep(15 * (attempt + 1))
                                            else:
                                                raise e
                                    time.sleep(1)
                                    
                                progress_text.empty()
                                st.cache_resource.clear()
                                st.success(f"Successfully imported {len(docs)} document(s) from Google Drive!")
                                st.rerun()
                            else:
                                st.warning("Could not load any documents.")
                        except Exception as e:
                            st.error(f"Error importing from Google Drive: {e}")

    chain = setup_rag_chain()

    if chain is None:
        st.warning("⚠️ No Data Room found. Please make sure PINECONE_API_KEY is set in your .env file or Streamlit Secrets.")
    else:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "pdf_bytes" in message:
                    st.download_button(
                        label="📄 Download Answer as PDF",
                        data=bytes(message["pdf_bytes"]),
                        file_name=f"Data_Room_Answer_{idx}.pdf",
                        mime="application/pdf",
                        key=f"dl_{idx}",
                        on_click=db.log_action,
                        args=(st.session_state["username"], "DOWNLOAD_PDF")
                    )

        # React to user input
        if prompt := st.chat_input("What would you like to know about your documents?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Searching the Data Room..."):
                    docs = chain["retriever"].invoke(prompt)
                    context = "\n\n".join(doc.page_content for doc in docs)
                    
                    answer = chain["llm_chain"].invoke({"input": prompt, "context": context})
                    st.markdown(answer)
                    
                    db.log_query(st.session_state["username"], prompt, answer)
                    
                    with st.expander("View Source Snippets"):
                        for i, doc in enumerate(docs):
                            source = doc.metadata.get("source", "Unknown")
                            st.caption(f"**Source {i+1}: {source}**")
                            st.write(doc.page_content)
                            
                    pdf_bytes = generate_watermarked_pdf(prompt, answer, st.session_state["username"])
                    
                    dl_key = f"dl_{len(st.session_state.messages)}"
                    st.download_button(
                        label="📄 Download Answer as PDF",
                        data=bytes(pdf_bytes),
                        file_name="Data_Room_Answer.pdf",
                        mime="application/pdf",
                        key=dl_key,
                        on_click=db.log_action,
                        args=(st.session_state["username"], "DOWNLOAD_PDF")
                    )
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "pdf_bytes": pdf_bytes
            })

import os
from langchain_google_community import GoogleDriveLoader
loader = GoogleDriveLoader(
    document_ids=["1Xy_..._abc"], 
    credentials_path=os.path.abspath("credentials.json"), 
    token_path=os.path.abspath("token.json")
)
try:
    loader.load()
except Exception as e:
    import traceback
    traceback.print_exc()

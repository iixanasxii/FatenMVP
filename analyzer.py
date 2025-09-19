"""
analyzer.py

This script analyzes (indexes) all files in the "docs/" folder,
and stores them into a local vectorstore ("db_store" folder).
Run it each time you create/edit/add a file in docs/.

Dependencies:
  langchain_community, langchain_nomic, etc.
"""

import os

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings

# Paths/config
PERSIST_DIRECTORY = "db_store"  # where the vectorstore is saved
DOCS_DIRECTORY = "docs"         # where your local .txt or .pdf files live

def analyze_files():
    """
    Load documents from DOCS_DIRECTORY, split them, and build a Chroma vectorstore,
    storing it in PERSIST_DIRECTORY. Overwrites any old data.
    """
    print(f"Analyzing documents in '{DOCS_DIRECTORY}'...")
    if not os.path.isdir(DOCS_DIRECTORY):
        print(f"Docs folder '{DOCS_DIRECTORY}' not found. Creating empty folder.")
        os.makedirs(DOCS_DIRECTORY, exist_ok=True)

    # Load local docs (all *.txt, or adjust as needed)
    loader = DirectoryLoader(DOCS_DIRECTORY, glob="*.txt")
    docs_list = loader.load()

    # Split docs
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_splits = splitter.split_documents(docs_list)

    # Create embeddings
    embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

    print(f"Building Chroma vectorstore in '{PERSIST_DIRECTORY}' ...")
    # Create (or overwrite) Chroma DB
    vectordb = Chroma.from_documents(
        documents=doc_splits,
        embedding=embedding_function,
        collection_name="my_collection",
        persist_directory=PERSIST_DIRECTORY,
    )
    print("Done! Vectorstore created/updated.")

if __name__ == "__main__":
    analyze_files()

""" use langchain to build a semantic search tool that reads through pdfs """
## reference : https://python.langchain.com/docs/tutorials/retrievers/
import json, os, hashlib
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# load env files
load_dotenv()

# set Dir paths
INDEX_DIR = "faiss_index"
HASH_FILE = "embedded_hashes.json"
DOCS_DIR = "docs"


def file_hash(file_path:str):
    """ return SHA256 hash of a file's contents. """
    # create hash
    hash = hashlib.sha256()
    
    # open and read file in binary
    with open(file_path, "rb") as file:
        hash.update(file.read())
    return hash.hexdigest()

def embded_store(llm_model="text-embedding-3-small") -> None:
    """ embed document and store using FAISS"""
    # define embedding odel
    embeddings = OpenAIEmbeddings(model=llm_model)

    # text splitter : define chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )

    # load previous hashes path if it exists
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as file:
            embedded_hashes = json.load(file)
    else:
        embedded_hashes = {}
    
    # load Faiss index if it exists
    if os.path.exists(INDEX_DIR):
        vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        # define vector store
        embedding_dim = len(embeddings.embed_query("hello")) # get vector size
        index = faiss.IndexFlatL2(embedding_dim)
        vector_store = FAISS(
            embedding_function = embeddings,
            index = index,
            docstore = InMemoryDocstore(),
            index_to_docstore_id = {},
        )
    
    # process docs
    for docname in tqdm(os.listdir(DOCS_DIR)):
        doc_path = os.path.join(DOCS_DIR, docname)
        if not os.path.isfile(doc_path):
            continue
        
        # get file hash
        doc_hash = file_hash(doc_path)

        # check if document embedding exists
        if embedded_hashes.get(docname) == doc_hash:
            print(f"File: {docname} already embedded")
            continue
        
        print(f"Embedding file : {docname}")

        # load and split doc
        loader = PyPDFLoader(doc_path)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)

        # vector store
        vector_store.add_documents(splits)
        embedded_hashes[docname] = doc_hash
    
    # save updated vector store and update hash to track embedded documents
    vector_store.save_local(INDEX_DIR)
    with open(HASH_FILE, "w") as hash_file:
        json.dump(embedded_hashes, hash_file)
    
    print("Index updated successfully")



# create and run prompt
prompt = """
    You're a lab automation expert helping an Automation Engineer set paramaters based on the questions they have.
    Use the embedded Liquid Handling reference guide to answer the question the engineer might have, if not covered in the guide respond with "Out of scope".
"""


if __name__ == "__main__":
    _ = embded_store()
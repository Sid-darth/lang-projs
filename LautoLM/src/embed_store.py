""" embed and store embedding using FAISS through langchain - store embedding index """
import os, json, hashlib
from dotenv import load_dotenv
import faiss
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, csv_loader, JSONLoader
# from langchain_docling import DoclingLoader
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# load env files for creds
load_dotenv()

# set dir paths
INDEX_DIR = "docs/faiss_index"
HASH_FILE = "docs/embedded_hashes.json"
DOCS_DIR = "docs/source"


def file_hash(file_path: str):
    """ return readable hash for input file """
    # create hash
    hash = hashlib.sha256()

    # open and read file in binary
    with open(file_path, "rb") as file:
        hash.update(file.read())
    return hash.hexdigest()

def create_vector_store(llm_model="text-embedding-3-small"):
    """ create vector store object """
    # load faiss index if embedded docs present
    if os.path.getsize(INDEX_DIR) > 0: 
        vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    
    # define embedding model
    embeddings = OpenAIEmbeddings(model=llm_model)
    embedding_dim = len(embeddings.embed_query("hello")) # get vector size
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function = embeddings,
        index = index,
        docstore = InMemoryDocstore(),
        index_to_docstore_id = {}
    )
    return vector_store

def load_split(doc_path: str) -> list:
    """ define text splitter based on doc ext and return list of chunks """
    # get doc extension
    doc_ext = doc_path.split(".")[-1:]

    # define text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50) # recursive splitter keeps larger units intact

    # if doc_ext == "pdf":
    #     loader = PyPDFLoader(doc_path)
    # elif doc_ext == "csv":
    #     loader = csv_loader.CSVLoader(file_path=doc_path)
    # elif doc_ext == "json":
    #     loader = JSONLoader(file_path=doc_path, text_content=False)
    # else:
    #     loader = DoclingLoader(file_path=doc_path)

    loader = PyPDFLoader(doc_path)
    docs = loader.load()
    split_list = text_splitter.split_documents(docs)

    return split_list


def embed_store(llm_model="text-embedding-3-small") -> None:
    """ embed text based document and store using FAISS """
    # define embedding model

    # load hash representations for exisiting embeddings if available
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as file:
            embedded_hashes = json.load(file)
    else:
        embedded_hashes = {}
    
    # load or create vector store
    vector_store = create_vector_store()
    
    # process docs
    for docname in tqdm(os.listdir(DOCS_DIR)):
        
        doc_path = os.path.join(DOCS_DIR, docname)
        if os.path.isfile(doc_path) is False:
            continue

        # get hash file
        doc_hash = file_hash(doc_path)
        print(f"docname : {doc_hash}")

        # check if document embedding exists
        if embedded_hashes.get(docname) == doc_hash:
            print(f"File: {docname} already embedded")
            continue
        
        print(f"Embedding file : {docname}")
        
        # load and split into chunks
        splits = load_split(doc_path)

        # vector store
        vector_store.add_documents(splits)

        # add doc to hash object to track embedded docs
        embedded_hashes[docname] = doc_hash

    # save updated vector store and update hash json
    vector_store.save_local(INDEX_DIR)
    with open(HASH_FILE, "w") as hash_file:
        json.dump(embedded_hashes, hash_file)
    
    print("Index updated successfully")

if __name__ == "__main__":
    _ = embed_store()
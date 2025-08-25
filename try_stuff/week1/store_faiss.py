"""
- embed text using sentence trnasformer
- store embedding in FAISS
- run a similarity search
"""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# declare model
model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_store(str_list: list) -> str:
    """ embed and store in faiss"""
    # gen embeddings
    embeddings = model.encode(str_list)
    embeddings = np.array(embeddings).astype("float32")

    # create faiss index (L2 similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # add embeddings to index and store index
    index.add(embeddings)
    doc_name = "dump_faiss.faiss"
    faiss.write_index(index, doc_name)
    print(f"Added {index.ntotal} documents to index at doc")
    
    return doc_name

def query_index(query:str, index_doc:str, k=2) -> str:
    """ create query vecotr and get similarity against created faiss index """
    # create vector embedding of the query
    query_vector = model.encode([query]).astype("float32")

    # search top 2 results
    index = faiss.read_index(index_doc)
    # get disntances and indices
    d, i = index.search(query_vector, k=k)
    print(f"Dist : {d}, Indices: {i}")

    return d, i




if __name__ == "__main__":
    test_doc = [
        "Ollama is a local model runner.",
        "Hugging Face provides pretrained embedding models.",
        "FAISS is a library for similarity search on dense vectors.",
        "This is a test document about embeddings."
    ]

    # store faiss index
    index_doc = encode_store(test_doc)

    # query and get similarity
    query = "How do I run models locally?"
    distances, indices = query_index(query, index_doc)

    for idx,dist in zip(indices[0], distances[0]):
        print(f"Result: {test_doc[idx]} (distance={dist:.4f})")
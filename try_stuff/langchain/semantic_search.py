from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"

# load environment variables
load_dotenv()

# define embedding model
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

# retrive faiss vector stores
vector_store = FAISS.load_local(
    INDEX_DIR,
    embeddings,
   allow_dangerous_deserialization=True
)


# run similarity search on user query
query = input("Enter query: ")
sim_docs = vector_store.similarity_search(query, k=5)

# combine retrieved docs
context = "\n\n".join([f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in sim_docs])

# define prompt template and create prompt
template = """
    You are an Automation Engineer experienced with programming, operating, and troubleshooting all things related to Hamilton Liquid handlers.
    Your knowledge encompasses the knowledge obtained from the documents, if you don't know say so but offer additional context you can find from the internet using forums if possible

    Context:
    {context}

    Question:
    {question}

    Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# get answer from llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
final_prompt = prompt.format(context=context,question=query)
response = llm.invoke(final_prompt)

print(response)
print("\n\n\n\n")
print(response.content)
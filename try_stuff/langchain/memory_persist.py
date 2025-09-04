""" script to test saving conversational memory when using langchain """
import json, os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict


# define file locations
MEMORY_FILE = "chat_memory.json"
INDEX_DIR = "faiss_index"

# load environment variables
load_dotenv()


def load_chat_history(mem_file:str = MEMORY_FILE)-> list:
    """ load chat history from json file"""
    if os.path.exists(mem_file):
        with open(mem_file, "r") as mf:
            message_data = json.load(mf)
            return message_data
    return []

# create function to create prompt and call it as part of the chain
def create_prompt()-> str:
    """ create prompt template that utilises chat history """
    
    system_template = """
        You're an Automation Engineer tasked with providing answers to user queries.
        Use the conversation history to maintain context and also take in user suggestions when made.
    """

    custom_prompt = PromptTemplate(
        input_variables = ["context", "chat_history", "question"],
        template = system_template + """
            Chat History:
            {chat_history}
            
            Context from documents:
            {context}

            Question: {question}

            Answer
        """
    )

    return custom_prompt

# define embedding model
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

# retrive faiss vector stores
vector_store = FAISS.load_local(
    INDEX_DIR,
    embeddings,
   allow_dangerous_deserialization=True
)

# # load previous chat memory if it exists
# if os.path.exists(MEMORY_FILE):
#     with open(MEMORY_FILE, "r") as mf:
#         messages = json.load(mf)
# else:
#     messages = []

# add previous memory if found to conversation context
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=3
)
messages = load_chat_history()
if len(messages) == 0:
    memory.chat_memory.messages = []
else:
    memory.chat_memory.messages = messages_from_dict(load_chat_history())

# define llm and memory parameters
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# define retriever
retriever = vector_store.as_retriever(search_kwargs={"k":3})

# create chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": create_prompt()},
    # return_source_documents=True
)

"""
# run queries through the chain
query = "What is contact angle?"
result = qa_chain.invoke({"question": query})
print(result)

# tune response
memory.chat_memory.add_user_message("Make sure to add reference info in all answers.")
memory.chat_memory.add_ai_message("Understood. I will include reference info in future answers.")

query = "what is vapor pressure?"
result = qa_chain.invoke({"question": query})
print(result)
"""

# test to run additional query after retrieving saved chat history
query = "What parameters should I consider fine tuning for a 50uL transfer liquid with high viscosity"
result = qa_chain.invoke({"question":query})

print(result)

# save messages so that they can be retrieved by the next run
with open(MEMORY_FILE, "w") as mf:
    json.dump(messages_to_dict(memory.chat_memory.messages), mf)



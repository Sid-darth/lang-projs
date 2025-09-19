""" chain conversations using langchain and call system prompt """
import json, os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict


# define file locations
MEMORY_FILE = "docs/chat_memory.json"

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
        Limit your response to a few sentences to directly answer the user question.

        Add a bullet point with the reference used at the end of the response
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

def save_chat_memory(memory, memory_file, limit=10):
    """ add conversation memory to last 10 entries """
    if os.path.exists(memory_file):
        with open(memory_file, "r", emcoding="utf-8") as f:
            try:
                existing_history = json.load(f)
            except json.JSONDecodeError:
                existing_history=[]
    else:
        exisiting_history=[]
    
    # limit retrieved entries
    truncated_history = exisiting_history[-limit:]

    # get current messages
    chat = messages_to_dict(memory.chat_memory.messages)

    # append current conversation
    updated_history = truncated_history+chat

    # reqwrite memory file
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(updated_history, f, indent=2, ensure_ascii=False)

def create_chain(
        input_query:str,
        vector_store,
        lang_model:str="gpt-4o-mini",
        instrument="hamilton",
        general = False,
        min_confidence=0.3,
        ) -> str:
    """ create langchain """

    # add previous memory to conversation context
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5
    )
    messages = load_chat_history()
    if len(messages) == 0:
        memory.chat_memory.messages = []
    else:
        memory.chat_memory.messages = messages_from_dict(load_chat_history())
    
    # define llm and memory parameters
    llm = ChatOpenAI(model=lang_model, temperature=0.1)

    # define retriever
    # list of keys to retrieve
    key_list = [{"doc_key":instrument}]
    if general == "Yes":
        key_list.append({"doc_key":"general"})
    retriever = vector_store.as_retriever(search_kwargs={
        "k":5, "filter": {
            "$or": key_list
        }
        }
    )

    # filter docs based on minimum confidence
    docs_with_scores = vector_store.similarity_search_with_score(
        input_query, k=5, filter={"$or": key_list}
    )

    docs_with_conf = [(doc, 1/(1+float(score))) for doc, score in docs_with_scores]
    print("conf:: ",docs_with_conf)
    docs_with_conf.sort(key=lambda x: x[1], reverse=True)

    filtered_docs = [doc for doc, conf in docs_with_conf if conf >= min_confidence]
    print(f'filtered_docs: {filtered_docs}')

    # message when nodocs meet confidence interval
    if not filtered_docs:
        return "No confidence in provided docs"

    # create chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": create_prompt()},
        # return_source_documents=True
    )

    result = qa_chain.invoke({"question":input_query})

    # write to memory file
    # save messages so that they can be retrieved by the next run
    with open(MEMORY_FILE, "w") as mf:
        json.dump(messages_to_dict(memory.chat_memory.messages), mf, indent=2, ensure_ascii=False)

    return result["answer"]

if __name__ == "__main__":

    # define embedding model
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

    INDEX_DIR = "docs/faiss_index"


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
    retriever = vector_store.as_retriever(search_kwargs={
        "k":3, "filter": {
            "$or":[
                {"doc_key": "general"},
                {"doc_key": "hamilton"}
            ]
        }
        }
    )

    # create chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": create_prompt()},
        # return_source_documents=True
    )


    # run queries through the chain
    query = "What is contact angle?"
    result = qa_chain.invoke({"question": query})
    print("X"*10)
    print(result)

    # tune response
    memory.chat_memory.add_user_message("Make sure to add reference info in all answers.")
    memory.chat_memory.add_ai_message("Understood. I will include reference info in future answers.")

    memory.chat_memory.add_user_message("All proceeding questions will relate to Hamilton liquid handling systems")
    memory.chat_memory.add_ai_message("Understood. I will answer questions with consideration for Hamilton liquid handling systems.")
    print("X"*10)
    print(memory.chat_memory.messages)
    print("X"*10)
    print("X"*10)
    """
    query = "what is vapor pressure?"
    result = qa_chain.invoke({"question": query})
    print(result)
    """

    # test to run additional query after retrieving saved chat history
    query = "What parameters should I consider fine tuning liquid class for a 50uL transfer liquid with high viscosity"
    result = qa_chain.invoke({"question":query})

    print(result)

    # save messages so that they can be retrieved by the next run
    with open(MEMORY_FILE, "w") as mf:
        json.dump(messages_to_dict(memory.chat_memory.messages), mf, indent=2, ensure_ascii=False)
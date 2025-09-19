""" create gradio app for chat tool """
import gradio as gr
import time
import random
from src.convo_chain import create_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

#### use Interface
#######
# def prompt_response(user_prompt:str):
#     return "Generating prompt respnse here"

# input_section = gr.Interface(
#     fn = prompt_response,
#     inputs = ["text"],
#     outputs = ["text"]
# )

# input_section.launch()
########


# load FAISS index
# define embedding model
embedding_model:str="text-embedding-3-small"
embeddings = OpenAIEmbeddings(model = embedding_model)
INDEX_DIR = "docs/faiss_index"

# retrieve faiss vector stores
vector_store = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

with gr.Blocks() as demo:
    # set markdown text
    gr.Markdown(
        """ Select instrument and type query <br>
        Click enter to get response"""
    )

    # Dropdown for instrument selection
    instrument_dropdown = gr.Dropdown(
        ["Hamilton", "Artel"],
        label="Select Instrument",
        value="Hamilton",  # Default selection
    )

    radio_general = gr.Radio(["Yes", "No"], label="Include general contextual information")
    print(f"RADIO: {radio_general}")

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        history = history or []
        history.append((user_message,None))
        # return empty textbox and updated history
        return "", history

    def bot(history, instrument_dropdown, radio_general):
        user_message = history[-1][0]
        bot_message = create_chain(user_message, vector_store, instrument=instrument_dropdown, general=radio_general)
        history[-1] = (user_message, bot_message)
        return history

    msg.submit(
        user, [msg, chatbot], [msg, chatbot], queue=True
    ).then(
        bot, [chatbot, instrument_dropdown, radio_general], chatbot
    )
    clear.click(lambda:[], None, chatbot, queue=False)

demo.launch()
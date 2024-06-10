import streamlit as st  
from streamlit_chat import message
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from pinecone import Pinecone
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from dotenv import load_dotenv
load_dotenv()

os.environ["PINECONE_API_KEY"] = os.environ['PINECONE_API_KEY']

embeddings = OllamaEmbeddings(model="all-minilm:latest")
pc = Pinecone()
index_name = "rag-pdf"
index = pc.Index(index_name)

def retriever(input):
    input_em = embeddings.embed_query(input)
    result = index.query(vector=input_em, top_k=2, includeMetadata=True)
    texts = []
    for match in result["matches"]:
        if "metadata" in match and "text" in match["metadata"]:
            texts.append(match["metadata"]["text"])
    return " ".join(texts)

st.subheader("RAG-PDF Chatbot")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = Ollama(model = "gemma:2b")

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the questions based on the provided context only.
Please provide the most accurate response based on the question and if the answer is not present in the context, please respond with I don't know.""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            context = retriever(query)
            if len(st.session_state.requests) == 0:
                response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
            else:
                response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
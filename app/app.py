import streamlit as st

import logging

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

def main(query):

    graph = Neo4jGraph()
 
    print(graph.schema)

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)
    response = chain.invoke({"query": query})
    print(response)
    return response['result']

logger = logging.getLogger(__name__)
logging.basicConfig(filename='app.log', encoding='utf-8', level=logging.INFO)


# Page configuration
st.set_page_config(page_title="Graph Search Tool", page_icon="🌐")
st.header("`Graph Search Tool`")
st.info("`I am an Graph Search tool equipped to provide insightful answers by delving into, comprehending, \
        and condensing information from Graph Database.`")
# Hide Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Container setup
reply_container = st.container()
container = st.container()

submit_button = None
user_input = None
with container:
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("`Ask a question:`", key='input')
        submit_button = st.form_submit_button(label='Send ⬆️')
    
    # Response generation
    if submit_button and user_input:
        result = main(user_input)    
        st.info(result)
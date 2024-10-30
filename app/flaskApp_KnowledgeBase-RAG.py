from flask import Flask, request, jsonify
from neo4j import GraphDatabase
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import gunicorn

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)

# Neo4j connection setup
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Generative AI
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize LLM with Google Generative AI (Gemini)
llm = GoogleGenerativeAI(model="gemini-pro")

# Function to get the database schema
def get_schema():
    with driver.session() as session:
        labels_query = """
        CALL db.labels() YIELD label
        RETURN collect(label) as labels
        """
        result = session.run(labels_query).single()
        labels = result['labels'] if result else []

        rels_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN collect(relationshipType) as relationships
        """
        result = session.run(rels_query).single()
        relationships = result['relationships'] if result else []

        schema = {}
        for label in labels:
            props_query = f"""
            MATCH (n:{label})
            RETURN distinct keys(n) as properties
            LIMIT 1
            """
            result = session.run(props_query).single()
            properties = result['properties'] if result else []
            schema[label] = properties

        return {
            'node_labels': labels,
            'relationships': relationships,
            'node_properties': schema
        }

# Function to extract knowledge base
def extract_knowledge_base():
    with driver.session() as session:
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n, r, m
        """
        results = session.run(query)

        text_chunks = []
        for record in results:
            node1 = record['n']
            rel = record['r']
            node2 = record['m']

            if rel and node2:
                chunk = f"{dict(node1)} is {rel.type} {dict(node2)}"
            else:
                chunk = f"Node: {dict(node1)}"
            text_chunks.append(chunk)

        return text_chunks

# Create vector store
def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

# Setup RetrievalQA with LLM
def setup_rag(vector_store):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa_chain

# Initialize schema, vector store, and RAG (retrieval-augmented generation) chain
schema = get_schema()
text_chunks = extract_knowledge_base()
vector_store = create_vector_store(text_chunks)
qa_chain = setup_rag(vector_store)

# Flask route to handle questions
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data.get("query", "")

    # Include the schema context in the query
    schema_context = f"Database Schema: {schema}\n\n"
    full_query = schema_context + query

    # Get the answer from the RAG system
    response = qa_chain.run(full_query)

    return jsonify({"response": response})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)

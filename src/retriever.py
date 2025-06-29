import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_neo4j import Neo4jVector
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from vector_db.setup_vector_db import load_documents

####################################

def load_embedding_model(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

def load_vector__retriever(embedding_model, vector_stores_path='vectorstores/my_db'):
    return FAISS.load_local(
        folder_path=vector_stores_path, 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
    ).as_retriever()
    
def load_graph__retriever(embedding):
    return Neo4jVector.from_existing_graph(
        embedding=embedding,
        url=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD'),
        index_name='events', # Vector index name
        node_label="Event", # Relative node label
        text_node_properties=['name', 'full_text', 'description'], # Properties to be used to calculate embeddings and retrieve from the vector index.
        embedding_node_property='embedding' # Which property to store the embedding values to.
    ).as_retriever()

def load_keyword_retriever():
    documents = load_documents()
    return BM25Retriever.from_documents(documents)
    
def create_ensemble_retriever(vector_retriever, graph_retriever, keyword_retriever):
    return EnsembleRetriever(
        retrievers=[vector_retriever, graph_retriever, keyword_retriever],
        weights=[1.0, 1.0, 1.0]
    )
    
def get_history_retriever():
    embedding_model = load_embedding_model('intfloat/multilingual-e5-large')
    vector_retriever = load_vector__retriever(embedding_model)
    graph_retriever = load_graph__retriever(embedding_model)
    keyword_retriever = load_keyword_retriever()
    return create_ensemble_retriever(vector_retriever, graph_retriever, keyword_retriever)
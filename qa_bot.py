import os
from dotenv import load_dotenv

from langchain import hub
from langchain_neo4j import Neo4jVector
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from pprint import pprint

load_dotenv()

def load_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv('GEMINI_MODEL'),
        api_key=os.getenv('GOOGLE_API_KEY')
    )
    
def load_embedding(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

def load_vector_index(embedding, url, username, password):
    return Neo4jVector.from_existing_graph(
        embedding=embedding,
        url=url,
        username=username,
        password=password,
        index_name='tasks', # Vector index name
        node_label="Task", # Relative node label
        text_node_properties=['name', 'description', 'status'], # Properties to be used to calculate embeddings and retrieve from the vector index.
        embedding_node_property='embedding' # Which property to store the embedding values to.
    )
    
def create_qa_chain(llm, vector_index):
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    return create_retrieval_chain(
        combine_docs_chain=combine_docs_chain,
        retriever=vector_index.as_retriever()
    )
    
def run_qa_chain(qachain, query):
    query = {"input": query}
    return qachain.invoke(query)
    
def main():
    NEO4J_URL = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    
    llm = load_llm()
    embedding = load_embedding('intfloat/multilingual-e5-large')
    vector_index = load_vector_index(embedding, NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD)
    qa_chain = create_qa_chain(llm, vector_index)
    
    query = "How will recommendation service be updated?"
    response = run_qa_chain(qa_chain, query)
    pprint(response)

if __name__ == '__main__':
    main()
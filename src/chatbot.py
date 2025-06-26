import os
from dotenv import load_dotenv
from pprint import pprint

from langchain_neo4j import Neo4jVector
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever

from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def load_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv('GEMINI_MODEL'),
        api_key=os.getenv('GOOGLE_API_KEY')
    )

def load_embedding_model(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

def load_vector_stores(embedding_model, vector_stores_path='vectorstores/my_db'):
    return FAISS.load_local(
        folder_path=vector_stores_path, 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
    ).as_retriever()
    
def load_graph_stores(embedding):
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
    
def create_retriever(vectorstores, graphstores):
    return EnsembleRetriever(
        retrievers=[vectorstores, graphstores],
        weights=[0.5, 0.5]
    )
    
def create_qa_chain(llm, retriever):
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)
    
    return create_retrieval_chain(
        combine_docs_chain=combine_docs_chain,
        retriever=retriever
    )
    
def run_qa_chain(qa_chain, query):
    query = {'input': query}
    return qa_chain.invoke(query)

def main():
    llm = load_llm()
    embedding = load_embedding_model('intfloat/multilingual-e5-large')
    vectorstores = load_vector_stores(embedding)
    graphstores = load_graph_stores(embedding)
    retriever = create_retriever(vectorstores, graphstores)
    qa_chain = create_qa_chain(llm, retriever)
    
    while(True):
        print('========================================================================')
        query = input("Nhập câu hỏi của bạn (Nhập 'q' để thoát): ")
        if query.lower() == 'q':
            return
        response = run_qa_chain(qa_chain, query)
        pprint(response)
        

if __name__ == '__main__':
    main()
    
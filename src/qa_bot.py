import os
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def load_graph(url, username, password):
    return Neo4jGraph(
        url=url,
        username=username,
        password=password
    )
    
def load_llm(temperature=0.2):
    return ChatGoogleGenerativeAI(
        model=os.getenv('GEMINI_MODEL'),
        api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=temperature,
    )
    
def create_cypher_qa_chain(llm, graph):
    graph.refresh_schema()
    
    return GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True
    )
    
def main():
    NEO4J_URL = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    
    graph = load_graph(NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD)
    llm = load_llm()
    cypher_qa_chain = create_cypher_qa_chain(llm, graph)
    
    query = "How many tickets there are?"
    response = cypher_qa_chain.invoke(query)
    
    print(f"Response: {response}")
    
    
if __name__ == '__main__':
    main()
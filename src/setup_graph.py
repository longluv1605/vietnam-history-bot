import os
import json
import requests
from pprint import pprint
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph

load_dotenv()

def load_query(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['query'] 

def create_graph(url, username, password):
    return Neo4jGraph(
        url=url,
        username=username,
        password=password
    )
    
def create_dataset(graph, query):
    graph.query(
        query
    ) 
    
def main():
    NEO4J_URL = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

    # Create neo4j graph    
    graph = create_graph(NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD)
    print(f"-> Created Graph: {graph}")

    # Insert data into graph
    query = load_query('data/query.json')
    create_dataset(graph, query)
    pprint(f"-> Inserted query into graph:\n{query}")
    

if __name__ == '__main__':
    main()
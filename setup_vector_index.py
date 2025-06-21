import os
from dotenv import load_dotenv

from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def load_embedding(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)
    
def create_vector_index(embedding, url, username, password):
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
    
def main():
    NEO4J_URL = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    
    # Create vector index
    embedding = load_embedding('intfloat/multilingual-e5-large')
    vector_index = create_vector_index(embedding, NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD)
    print("-> Created vector index.")
    response = vector_index.similarity_search(
        "How will RecommendationService be updated?"
    )
    print("-> Retrieved:", response[0].page_content)
    # name: BugFix
    # description: Add a new feature to RecommendationService to provide ...
    # status: In Progress
    

if __name__ == '__main__':
    main()
from pprint import pprint

import pdf2image
import pytesseract

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document


def load_embedding_model(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

def load_documents(data_path='data/Lich su 12.pdf'):
    # Step 1: Load pages
    images = pdf2image.convert_from_path(data_path)
    
    # Step 2: use OCR to extract docs
    documents = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang='vie')
        documents.append(Document(page_content=text, metadata={"page": i+1}))
        
    return documents

def chunking(documents, chunk_size=1024, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=documents)
    
    return chunks
    

def create_simple_vector_stores(embedding_model, data_path, vector_stores_path="vectorstores/my_db"):
    # Load docs
    documents = load_documents()
    
    # Chunking
    chunks = chunking(documents)
    
    # Create vector stores
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_stores_path)
    
    return db

def main():
    data_path = 'data/Lich su 12-5-19.pdf'
    model_name = 'intfloat/multilingual-e5-large'
    
    embedding_model = load_embedding_model(model_name)
    vector_db = create_simple_vector_stores(embedding_model, data_path)
    
    docs = vector_db.similarity_search("Liên hợp quốc thành lập khi nào?")
    pprint(docs)

if __name__ == "__main__":
    main()
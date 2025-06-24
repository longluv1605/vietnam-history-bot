import os
from pprint import pprint
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate

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
    
def create_cypher_prompt():
    cypher_prompt = PromptTemplate.from_template("""
    Bạn là một chuyên gia về Neo4j và Cypher, có nhiệm vụ chuyển câu hỏi của người dùng sang truy vấn Cypher chính xác.

    # QUAN TRỌNG (IMPORTANT):
    - Chỉ trả lời bằng tiếng Việt (only Vietnamese).
    - KHÔNG dịch sang tiếng Anh.
    - KHÔNG giả định hoặc đổi tên thực thể.
    - Nếu không chắc tên chính xác, hãy dùng `CONTAINS`, `=~`, hoặc `STARTS WITH` để truy vấn mờ.
    - Chỉ sử dụng thông tin có trong schema bên dưới.
    - Luôn giới hạn kết quả bằng LIMIT nếu cần.

    # Schema:
    {schema}

    # Ví dụ:

    Q: Ai là người tham dự Hội nghị Ianta?
    Cypher:
    MATCH (e:Event)
    WHERE toLower(e.name) CONTAINS toLower("ianta")
    WITH e
    MATCH (e)-[:PARTICIPATED_IN]->(p:Person)
    RETURN p.name

    Q: Liên Hợp Quốc được thành lập trong sự kiện nào?
    Cypher:
    MATCH (e:Event)
    WHERE toLower(e.name) CONTAINS toLower("Liên Hợp Quốc")
    RETURN e.full_text

    # Câu hỏi:
    {question}

    Cypher:
    """
    )
    return cypher_prompt
    
def create_cypher_qa_chain(llm, graph, cypher_prompt):
    graph.refresh_schema()
    
    return GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=graph,
        verbose=True,
        validate_cypher=True,
        cypher_prompt=cypher_prompt,
        allow_dangerous_requests=True
    )
    
def main():
    NEO4J_URL = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    
    llm = load_llm()
    graph = load_graph(NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD)
    prompt = create_cypher_prompt()
    cypher_qa_chain = create_cypher_qa_chain(llm, graph, prompt)
    
    while(True):
        print('========================================================================')
        query = input("Nhập câu hỏi của bạn (Nhập 'q' để thoát): ")
        if query.lower() == 'q':
            return
        response = cypher_qa_chain.invoke(query)
        pprint(response)
    
    
if __name__ == '__main__':
    main()
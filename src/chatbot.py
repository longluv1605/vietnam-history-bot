import os
from dotenv import load_dotenv
from pprint import pprint

from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.prompts import PromptTemplate

from retriever import get_history_retriever

################################################

load_dotenv()

def load_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv('GEMINI_MODEL'),
        api_key=os.getenv('GOOGLE_API_KEY')
    )
    
def create_prompt_template():
    template = """
    [system]
    Bạn là một trợ lý chuyên giải đáp các thắc mắc về môn lịch sử trung học phổ thông (cụ thể là lớp 12, tại Việt Nam).
    Các câu trả lời của bạn phải liên quan đến bộ môn lịch sử và dựa vào thông tin được cung cấp.
    Bạn chỉ được trả lời bằng tiếng Việt.
    Nếu câu hỏi không liên quan đến môn lịch sử, hãy trả lời rằng bạn không trả lời các câu hỏi ngoài môn học.
    
    [user]
    Sau đây là câu hỏi của người dùng: {input}.
    
    [chat history]
    Đây là lịch sử chat của bạn và user: {chat_history}.
    
    [context]
    Đây là thông tin được cung cấp: {context}.
    """
    return PromptTemplate.from_template(template)


def create_qa_chain(llm, retriever):
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retrieval_qa_chat_prompt = create_prompt_template()
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)
    
    return create_retrieval_chain(
        combine_docs_chain=combine_docs_chain,
        retriever=retriever
    )
    
def run_qa_chain(qa_chain, chat_history, query):
    query = {
        'input': query,
        'chat_history': chat_history.messages,
    }
    return qa_chain.invoke(query)

def update_chat_history(chat_history, user_message, ai_message):
    chat_history.add_user_message(user_message)
    chat_history.add_ai_message(ai_message)
    return chat_history

def main():
    llm = load_llm()
    retriever = get_history_retriever()
    qa_chain = create_qa_chain(llm, retriever)
    chat_history = InMemoryChatMessageHistory()
    
    while(True):
        print('========================================================================')
        query = input("Nhập câu hỏi của bạn (Nhập 'q' để thoát): ")
        if query.lower() == 'q':
            return
        response = run_qa_chain(qa_chain, chat_history, query)
        answer = response['answer']
        print(f"Câu trả lời: {answer}\n")
        
        chat_history = update_chat_history(chat_history, query, answer)
        

if __name__ == '__main__':
    main()
    
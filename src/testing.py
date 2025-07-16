import sys
from chatbot import *
import time

def main(): 
    test_file = sys.argv[1]
    
    llm = load_llm()
    retriever = get_history_retriever()
    qa_chain = create_qa_chain(llm, retriever)
    chat_history = InMemoryChatMessageHistory()
    
    
    with open(test_file, 'r', encoding='utf-8') as f:
        questions = f.read().split('\n\n')
        
        true_answers = questions[-1].split('\n')[1].split(' ')
        answers = []
        instruction = f'''
        Sau đây tôi sẽ hỏi câu hỏi trắc nghiệm, bạn chỉ cần đưa ra đáp án.
        Ví dụ: "Câu 0. Ai đang đi bộ? A. Long B. Nam C. Lê D. Thắng"
        Thì bạn sẽ trả lời là: "A".
        Nhớ kỹ, chỉ đưa ra ký tự đại diện cho đáp án (A, B, C, D).\n
        '''
        
        questions = questions[:-1]
        for question in questions:
            query = instruction + question
            response = run_qa_chain(qa_chain, chat_history, query)
            answer = response['answer']
            answers.append(answer)
            print(question)
            print(answer)
            print('----------------')
            time.sleep(10)
    
    print(f"Đáp án đúng:    {true_answers}")
    print(f"Đáp án của bạn: {answers}")
    
    correct = sum(int(answers == true_answers))
    print(f"Số câu đúng: {correct}/{len(true_answers)} -> Điểm = {correct*10.0/len(true_answers):.2f}")
    
if __name__ == '__main__':
    main()
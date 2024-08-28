import os
import sys
import time

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# OpenAI API key 로드
# 팀리트리버 API KEY = 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

# 현개인 API KEY = sk-4gTdF4JQMisziASq9mz9Cuyxm4_Xq-VyfHw3IB3QN_T3BlbkFJN2Ct1hUYi3CiBfRi_rDQpDqj3FWAmsT07TifnQKzsA

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

#if not openai_api_key:
#    raise ValueError("API key not found. Please check your .env file or provide a valid API key.")

# OpenAI ChatGPT model : gpt-4 설정
llm = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4o",  # 사용할 GPT 모델
    temperature=0,
    max_tokens=1000
)

# 수의사 프롬프트 템플릿 정의
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    당신은 강아지의 건강, 특히 강아지의 변에 대해 아주 특화되어 변을 보고 강아지의 상태를 잘 파악하는 경험 많은 수의사입니다.
    강아지의 변을 보고, 변 상태에 관한 다음 질문들에 대해 상세하고 정확하게, 그리고 전문적인 답변을 제공해주세요.

    질문 : {question}

    문서 내용:
    {context}

    수의사로서의 답변 :
    """
)

#! chain.py
# LLMChain 생성 (chain.py)
qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)

# 문서 로드 및 벡터화
documents = []
doc_folder = "/Users/hyunowk/Downloads/ddong_chat/docs/txt"

for file in os.listdir(doc_folder):
    file_path = os.path.join(doc_folder, file)
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    else:
        continue

'''
# 로드된 문서 확인
if not documents:
    print("로드된 자료가 없습니다.")
else:
    print(f"Loaded {len(documents)} documents")
    for doc in documents:
        print(doc)
'''


#! split.py
# chunk size 1000, chunk overlap 10 으로 문서 분할
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

#! vectorstore.py
# Vector DB 생성 및 문서 삽입
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory="./data")
vectordb.persist()


# 질의응답 체인 생성 (수의사 프롬프트 템플릿 적용)
poopy_qa = ConversationalRetrievalChain(
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    combine_docs_chain=qa_chain,
    return_source_documents=True,
    verbose=False
)

# 챗봇 인터페이스
yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the Poopy Chatbot. 강아지똥?? 다 물어봐~')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query in ["exit", "quit", "q", "f"]:
        print('Exiting')
        sys.exit()
    if not query.strip():
        continue
    
    # 응답 시간 측정 - 시작
    #chat_start_time = time.time()
    
    result = poopy_qa({"question": query, "chat_history": chat_history})
    
    # 응답 시간 측정 - 종료
    #chat_end_time = time.time()

    #elapsed_time = chat_end_time - chat_start_time
    
    print(f"chat history is : {chat_history}")
    print(f"{white}Answer: " + result["answer"])
    #print(f"Response Time: {elapsed_time:.4f} 초 소요됨 (확인용)")  # 응답 시간 출력 (확인용!!)
    
    chat_history.append((query, result["answer"]))
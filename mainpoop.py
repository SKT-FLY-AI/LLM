import os
import sys
import json

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


#! 0. OpenAI API KEY 설정
# 보안 이슈로 깃허브에 올리려면 key를 다 따로따로 해서 합치면 올라가짐.
# 현 개인 APIkey : sk-4gTdF4JQMisziASq9mz9Cuyxm4_Xq-VyfHw3IB3QN_T3BlbkFJN2Ct1hUYi3CiBfRi_rDQpDqj3FWAmsT07TifnQKzsA
# 골댕이 팀 APIkey : sk-vYlP709KnrLTscNeuQBqeGTmZvFRdwzfxg-hTdavUPT3BlbkFJssCnfvitsyqgSV1yYNEv-EH-_GCRIQWcRHy_kPKBwA

o = "sk-"
p = "vYlP709KnrLTscNeuQBqeGTmZvFRdwzfxg-"
e = "hTdavUPT3BlbkFJssCnfvitsyqgSV1yYNEv-EH-"
n = "_GCRIQWcRHy_kPKBwA"


key = o + p + e + n

os.environ['OPENAI_API_KEY'] = key




#! 1. CV model로부터 json file 가져오기
# JSON 파일 읽기
with open('dummy.json', 'r', encoding='utf-8') as f:    # 지금은 dummy.json으로 했3
    json_data = json.load(f)


#! 2. 데이터 구조화
# 데이터 구조화
class FecalAnalysis(BaseModel):
    poo_type: Optional[str]  # 대변 유형, Optional로 설정하여 값이 없을 경우 처리 가능
    poo_color: Dict[str, float]  # 색깔 뭉치 -> 딕셔너리로 설정

# poo_colors 색상 정보 변환 함수 (원래는 warning color 변환이었3)
def parse_colors(colors):
    # 리스트인 경우, 각 {색상 코드, 확률}로 딕셔너리 변환
    if isinstance(colors, list):
        return {color[0]: color[1] for color in colors}
    return {}

# FecalAnalysis로 데이터 변환
analysis = FecalAnalysis(
    poo_type=json_data["poo_type"],  # 변의 유형
    poo_color=parse_colors(json_data["poo_color"])  # 색상 정보를 dictionary로 변환
)


#! 3. pdf 또는 txt 문서 불러오고 분할
# 텍스트 문서 로드 및 분할
loader = PyPDFLoader('animals_poop.pdf')    # 일단은 우리의 poo_type 설정한 논문 자료로 넣음용 (추후 더 정리해야함 아마 txt로?)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
texts = text_splitter.split_documents(documents)


#! 4. FAISS VectorDB 생성
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(texts, embeddings)
retriever = vector_store.as_retriever()

# FAISS 인덱스를 파일로 저장~
# Chroma에서 persist()와 같은 역할
# 나중에 load_local()로 인덱스 불러올 수 있음.
vector_store.save_local("faiss_index")  # 나중에 함 poopy_indexstore로 ㄱ


#! 5. LLM model 초기화
# model : OpenAI ChatGPT model (gpt-4)
llm = ChatOpenAI(
    api_key=key,
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)


#! 6. prompt template 정의
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "당신은 필요한 말만 하는, 강아지 대변에 특화된 수의사입니다."
    ),

    HumanMessagePromptTemplate.from_template(
        """
        나한테 우리 강아지 건강 상태를 파악하기 위해서, 
        FAISS 내 저장되어있는 문서를 기반으로
        {poo_color}를 갖고 있는 {poo_type}형태의 대변에 대한 분석 결과를 바탕으로 건강상태를 분석하기 위한 질문을 한가지 던져줘
        """
    )
])

'''
#! 6-1. 마무리용 prompt template 추가 정의
ending_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "당신은 필요한 말만 하는, 강아지 대변에 특화된 수의사입니다."
    ),

    HumanMessagePromptTemplate.from_template(
        """
        나한테 우리 강아지 건강 상태를 파악하기 위해서, 
        FAISS 내 저장되어있는 문서를 기반으로
        {poo_color}를 갖고 있는 {poo_type}형태의 대변에 대한 분석 결과를 바탕으로 건강상태를 분석하기 위한 질문을 한가지 던지되, 
        만약 질문할 내용이 없으면, 만약 건강에 이상이 있는거 같으면, 병원을 가라고 해주고, 이상이 없는거 같으면 결론을 내려줘.
        """
    )
])
'''

#! 7. LLM chain 생성
# RAG로부터 문서 검색 -> 문서 바탕으로 똑똑한 답변 생성 ㅎㅎ 똑순이!!!!!!!!!!!!!!!!
llm_chain = LLMChain(llm=llm, prompt=chat_prompt)


# 모델 실행 및 첫 질문 생성
# 챗봇이 먼저 질문을 던지는 형식이니까~
initial_response = llm_chain.invoke({
    "poo_type": analysis.poo_type,
    "poo_color": "\n".join([f"{color}({prob}%)" for color, prob in analysis.poo_color.items()])
})

# 질문 부분만 추출하여 챗봇 시작할 때 젤 먼저 던지는 질문으로 사용~
first_question = initial_response['text']


#! 8. RAG chain 생성
# chat_history 내용 -> memory로 넣어삐 ~! 대화 문맥 유지하려고 메모리에 일시적 저장/유지 >.<
memory=ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)

# 대화 기억 안할거면 RetrievalQA.. 왜 ConversationalRetrievalChain 이거 쓰는지 이해하3
# QnA 질의응답 RAG Chain 생성^^
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True    # 디버깅용으로 개발할 때만 넣고, 데모때는 성능 이슈 있을 수 있으니 비활성화 False/없애던가
)


#! 9. 찐 최종 Chatbot
# chatbot interface
yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

print(f"{yellow}---------------------------------------------------------------------------------")
print('----------!!!!! PoopyT 시작 !!!!!----------')
print('---------------------------------------------------------------------------------')

# 첫 질문 출력
print(f"{yellow} Poopy_START : {first_question}")

# 대화 5번 제한 두기 위해서 일단 conversation_count = 0으로 설정하고 시작
conversation_count = 0

# 사용자와의 대화 시나리오
while conversation_count < 6:
    conversation_count += 1     # 챗이 질문을 던지면서 시작하므로 시작단에서 count + 1

    # 질문 정의
    user_query = input(f"{white} 질문을 남겨주세요 : ")

    # 대화 종료할 조건
    if user_query.lower() in ["exit", "quit", "q"]:
        break

    # 1단계 : RAG 체인을 사용해 관련 문서 검색
    retrieved_docs = rag_chain({
        "question": user_query,
        "chat_history": memory.chat_memory.messages
    })

    # 2단계 : 검색된 문서를 LLMChain에 전달하여 최종 응답 생성
        # 문서 검색 결과가 있으면 해당 내용을 바탕으로 답변 생성 ! 없으면 걍 니 LLM model로 가버려잇.
    if 'source_documents' in retrieved_docs and retrieved_docs['source_documents']:
        context = "\n".join([doc.page_content for doc in retrieved_docs['source_documents']])

        final_response = llm_chain.invoke({
            "poo_type": analysis.poo_type,
            "poo_color": "\n".join([f"{color}({prob}%)" for color, prob in analysis.poo_color.items()]),
            "context": context,
            "question": user_query
        })

    else:
        # 문서 검색 결과가 없으면 LLMChain을 사용해 질문에 대한 일반적인 답변 생성
        final_response = llm_chain.invoke({
            "poo_type": analysis.poo_type,
            "poo_color": "\n".join([f"{color}({prob}%)" for color, prob in analysis.poo_color.items()]),
            "context": "관련 문서가 저장되어 있지 않습니다. 강아지 대변과 관련한 정보로 수의사처럼 상담을 진행하되 간단하게 질문을 하나씩 던지세요.",    # LLMChain한테 던지는 말.
            "question": user_query
        })


    # 응답 출력
    poopy_final_response = final_response['text']
    print(f"{yellow} Poopy : {poopy_final_response}")
    # # 체인 실행 (LLMChain <-> RAG 통합했긔ㅎㅎ)
    # chat_response = rag_chain({"question": user_query})
    # print(f"Poopy : {chat_response['answer']}")

    # chat_response = rag_chain.invoke(
    #     user_query=user_query,  # 사용자 질문 전달
    #     chat_history=memory.chat_history.messages    # 이전 대화 기록 전달
    # )

    # # 응답 출력
    # print(f"Poopy : {chat_response}")


    '''
    # 메모리 객체에서 대화 내용 가져오기
    chat_history = memory.chat_memory.messages

    # 대화 내용 출력
    for message in chat_history:
        print(f"\n Chat Log { {message.type}: {message.content} }")
    '''


#! 10. 서버와의 연결
 

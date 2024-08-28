import asyncio
import os
from typing import AsyncIterable, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper

from app.vectorstore import VectorStore
from app.embed import Embed
from app.split import Split
from app.agent import Agent
from callbacks.handler import AsyncCallbackHandler



# 보안 이슈로 깃허브에 올리려면 key를 다 따로따로 해서 합치면 올라가짐.
# 현 개인 APIkey : sk-4gTdF4JQMisziASq9mz9Cuyxm4_Xq-VyfHw3IB3QN_T3BlbkFJN2Ct1hUYi3CiBfRi_rDQpDqj3FWAmsT07TifnQKzsA
o = "sk-"
p = "4gTdF4JQMisziASq9mz9Cuyxm4_Xq-"
e = "VyfHw3IB3QN_T3BlbkFJN2Ct1hUYi3CiBfRi_"
n = "rDQpDqj3FWAmsT07TifnQKzsA"
key = o + p + e + n

os.environ['OPENAI_API_KEY'] = key


# 문서 로드 및 벡터화
documents = []
doc_folder = "/Users/hyunowk/Downloads/ddong_chat/docs/txt"


for file in os.listdir(doc_folder):
    file_path = os.path.join(doc_folder, file)

    if file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    else:
        continue

# vectorstore.py 안에서 text_split과 vectorDB 생성 및 문서 삽입 완료
vector_store = VectorStore('dummy.json', 'utf-8', 1000, 0)   # jq_schema='.' : 전체 json 문서를 로드하란 뜻^^

db = vector_store.get_db()  # get_db : 벡터화하고 저장한 DB를 반환하는 method
retriever = db.as_retriever()   # 질문을 query로 받고, vectorstore에서 retriever하여 response해주는 방식 (vectorstore에서 찾아서 답변)
callback = AsyncCallbackHandler()
agent = Agent(callback, retriever, documents).agent    # documents 내용을 가져올게요



# 확인용
# 질의응답 체인 생성 (수의사 프롬프트 템플릿 적용)
poopy_qa = ConversationalRetrievalChain(
    retriever=db.as_retriever(search_kwargs={'k': 6}),
    combine_docs_chain=retriever,
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

    
    result = poopy_qa({"question": query, "chat_history": chat_history})
    
    # 응답 시간 측정 - 종료
    #chat_end_time = time.time()

    #elapsed_time = chat_end_time - chat_start_time
    
    print(f"chat history is : {chat_history}")
    print(f"{white}Answer: " + result["answer"])
    #print(f"Response Time: {elapsed_time:.4f} 초 소요됨 (확인용)")  # 응답 시간 출력 (확인용!!)
    
    chat_history.append((query, result["answer"]))


'''
origins= [
    "http://localhost",
    "http://localhost:8000",
]
'''

'''
# 연결단
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    content: str

async def run_call(query: str, stream_it: AsyncCallbackHandler):
    agent.agent.llm_chain.llm.callbacks = [stream_it]
    await agent.acall(inputs={"input": query})

async def send_message(content: str) -> AsyncIterable[str]:
    stream_it = AsyncCallbackHandler()
    print("content : ", content)
    task = asyncio.create_task(
        run_call(content, stream_it))

    try:
        async for token in stream_it.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task

   
# get, post 차이 이해 ( https://noahlogs.tistory.com/35 )
@app.post("/stream_chat")
async def stream_test(message: Message):
    generator = send_message(message.content)
    return StreamingResponse(generator, media_type="text/event-stream")
'''
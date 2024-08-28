# chatbot이 생성한 답변을 비동기적으로 처리하는 handler 파일
from typing import Any
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import LLMResult

# (클래스 초기화 시 AsyncCallbackHandler를 상속받음)
class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False
    
    def __init__(self) -> None:
        super().__init__()
        self.final_flag = 0 # final_flag : 최종 응답을 표시하는 플래그
        #self.except_token = ['"', '}', "}", "\n}", "```"]   # 예외 처리 (이건 나중에 프론트와 할 때 생각해볼게연..)
        
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        if self.final_answer:
            if '"action_input": "' in self.content:
                for tk in self.except_token:
                    if tk in token:
                        return
                
                self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""
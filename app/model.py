from langchain.chat_models import ChatOpenAI

# LLM class : 초기화될 때 callback 함수, streaming , model 이름을 인수로 받음
class LLM:
    def __init__(self, callback, streaming=True, model_name="gpt-4"):
        self.llm = self.create_llm(callback, streaming, model_name)

    # create_llm 함수 : ChatOpenAI class를 통해 chatbot model 생성
    def create_llm(self, callback, streaming, model_name):
        return ChatOpenAI(
            streaming=streaming,    # streaming : 스트리밍이 활성화된 경우, 챗봇이 실시간으로 응답을 생성하고 전달함.
            model_name=model_name,  # model_name : 모델의 이름 지정
            callbacks=[callback],   # callbacks : 챗봇이 응답을 생성한 후, 호출할 콜백 함수 지정
        )
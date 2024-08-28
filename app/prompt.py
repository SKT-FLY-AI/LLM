from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

class Prompt:
    def __init__(self):
        self.prompt = self.create_prompt()

    def create_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                # StstemMessage : chatbot의 역할 & 행동방식 설명
                SystemMessage(
                    content=(
                        """
                        나한테 우리아이 건강 상태를 파악하기 위해서, 이 문서를 기반으로 질문을 한 가지만 해줘
                        질문할 내용이 없으면, 만약 건강에 이상이 있는거 같으면, 병원을 가라고 해주고 
                        이상이 없는거 같으면 결론을 내려주면 좋을거 같아
                        """
                    )
                ),
                # HumanMessagePromptTemplate : 사람 메시지 템플릿 - 사용자의 입력을 처리하는 방식을 {text}로 대체하여 처리
                HumanMessagePromptTemplate.from_template("{text}"),
            ]
        )
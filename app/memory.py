from langchain.memory import ConversationBufferWindowMemory

# ConversationBufferWindowMemory : 대화의 상호작용 유지를 위해 사용

class Memory:
    def __init__(self):
        self.memory = self.create_memory()

    def create_memory(self):
        return ConversationBufferWindowMemory(  # ConversationBufferWindowMemory : 대화의 상호작용 유지를 위해 사용
            memory_key="chat_history",
            k=5,    # 이 때, k개만큼의 interaction을 사용(기억)하여, buffer가 너무 커지지 않도록 함.
            return_messages=True,
            output_key="output"
        )
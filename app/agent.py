from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from app.memory import Memory
from app.chain import Chain
from app.prompt import Prompt
from app.model import LLM


# RAG with Agents - Retriever tool
    # 의료 정보가 담긴 txt 파일과, 개인 정보를 보고 잘 대답 하게 하기 위해서 description 작성
class Agent:
    def __init__(self, callback, retriever, json_file):
        self.llm = LLM(callback).llm
        self.memory = Memory().memory
        self.chain = Chain(self.llm, retriever, callback).chain
        self.prompt = Prompt().prompt
        self.tool = self.create_tool(retriever, json_file)
        self.agent = self.create_agent(self.tool)

    def create_tool(self, retriever, json_file):
        return create_retriever_tool(
            retriever,
            "data.csv",
            " "
        )

    def create_agent(self, tool):
        tools = [tool]


        # initialize_agent
            # AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION : JSONAgentOutputParser 형태
            # 프론트한테 추후 넘겨줄 때 사용하려고 하는.

        return initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=self.llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate",
            memory=self.memory,
            chain=self.chain,
            prompt=self.prompt,
        )
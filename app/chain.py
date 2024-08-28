from langchain.chains import ConversationalRetrievalChain
from app.model import LLM

class Chain:
    def __init__(self, llm, retriever, callback):
        self.llm = llm
        #self.llm = LLM(callback).llm
        self.chain = self.create_chain(self.llm, retriever)

    def create_chain(self, llm, retriever):
        return ConversationalRetrievalChain.from_chain_type(
            llm=llm,
            chain_type="refine",    # stuff, map_reduce, refine, map_rerank 중 refine은 대화의 맥락을 지속적으로 개선하고 유지할 때 사용
            retriever=retriever
        )
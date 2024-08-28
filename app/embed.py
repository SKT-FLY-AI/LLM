from langchain.embeddings.openai import OpenAIEmbeddings

class Embed:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
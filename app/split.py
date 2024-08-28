from langchain.text_splitter import RecursiveCharacterTextSplitter 

class Split:
    def __init__(self, chunk_size=1000, chunk_overlap=10, separators=[" ","\n",",","."]):    # chunk_size=1500, chunk_overlap=1
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=separators)


#! chunk size 1000, chunk overlap 10 으로 문서 분할
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
#documents = text_splitter.split_documents(documents)
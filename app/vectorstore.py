from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter


class VectorStore:
    def __init__(self, json_file_path, encoding, chunk_size, chunk_overlap, jq_schema=None):
        self.embeddings = OpenAIEmbeddings()    # OPENAI에서 제공하는 embedding model 사용
        # file_path=json_file_path cv로부터 받아오는 json 파일
        # JSON 파일 로더 생성 & 지정된 JSON 파일로부터 데이터 불러옴.
        #loader=JSONLoader(file_path=json_file_path, encoding=encoding, jq_schema='.', text_content=False)   # jq_schema='.', text_content=False
        loader=JSONLoader('dummy.json', jq_schema='.', text_content=False)

        docs=loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts=text_splitter.split_documents(docs)


        self.file_path = json_file_path
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.jq_schema = jq_schema


        # FAISS : 빠른 근접 이웃 검색 라이브러리인 FAISS를 통해 불러온 데이터를 벡터화하고, 이를 DB에 저장.
        self.db = FAISS.from_documents(texts, self.embeddings)

        

    # get_db : 벡터화하고 저장한 DB를 반환하는 method
    def get_db(self):
        return self.db
    

# cv model로부터 받아오는 json 파일 변환하여 생성된 query를 Vectorstore(FAISS)에서 실행

'''
def query_vector_store(query):
    vector_store = VectorStore('dummy.json', 'utf-8', 1000, 0)
    results = vector_store.search(query)
    return results
'''
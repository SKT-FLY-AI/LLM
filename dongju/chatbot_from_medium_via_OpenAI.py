import os
import sys
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser


# 당장은 OpenAI 사용 ^__^
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


documents = []
# Create a List of Documents from all of our files in the ./docs folder
for file in os.listdir("/mnt/sdb/tmp/poopsee/langchain/via_gpt/docs"):
    if file.endswith(".pdf"):
        pdf_path = "/mnt/sdb/tmp/poopsee/langchain/via_gpt/docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "/mnt/sdb/tmp/poopsee/langchain/via_gpt/docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "/mnt/sdb/tmp/poopsee/langchain/via_gpt/docs/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

# Convert the document chunks to embedding and save them to the vector store
try:
    vectordb = Chroma.from_documents(documents, embedding=OllamaEmbeddings(model="gemma2:27b"), persist_directory="./data")
    vectordb.persist()
except Exception as e:
    print(f"Error initializing Chroma: {e}")
vectordb.persist()

# create our Q&A chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    Ollama(model="llama3.1:70b"),
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = pdf_qa.invoke(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))#The LangChain repository
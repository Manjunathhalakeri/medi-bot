from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

#load raw pdfs

DATA_PATH="data/"
loader = DirectoryLoader(
    DATA_PATH,            # Directory to search
    glob="**/*.pdf",         # Pattern for PDF files in all subfolders
    loader_cls=PyPDFLoader   # Loader class for PDFs
)

documents = loader.load()

#chuunk the pdf's
#create embeddings OF THe chunks    
#store the embeddings in a vector store FAISS


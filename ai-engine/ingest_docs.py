from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

# Load and split files
loader = TextLoader("../data/sample_merchant_faqs.md")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Store in vector DB
Pinecone.from_documents(
    chunks,
    embeddings,
    index_name=os.getenv("PINECONE_INDEX_NAME")
)
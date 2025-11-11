
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")

os.environ["OPENAI_API_KEY"] = LLM_API_KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

loader = DirectoryLoader("docs", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(
    model="kwaipilot/kat-coder-pro:free",  # 👈 free model
    temperature=0.3,
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

query = input("Ask a question: ")
result = qa.invoke({"query": query})

print("\n🧠 Answer:\n", result["result"])
print("\n📄 Source Documents:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "unknown"))


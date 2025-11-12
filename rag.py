
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

os.environ["OPENAI_API_KEY"] = LLM_API_KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# --- 1️⃣  Load and preprocess documents ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists("faiss_index"):
    print("📂 Loading existing FAISS index...")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    print("🧠 Building new FAISS index...")
    loader = DirectoryLoader("docs", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")

# --- 2️⃣  Define LLMs ---
main_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3)

# --- 3️⃣  Build retrievers ---
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=main_llm,
)

# --- 4️⃣  Combine retrievers using Rank Reciprocal Fusion (RRF) ---
# weights determine how much each retriever contributes to the final result
ensemble_retriever = EnsembleRetriever(
    retrievers=[base_retriever, multi_retriever],
    weights=[0.5, 0.5],
)

# --- 6️⃣  RetrievalQA chain using RRF ---
qa = RetrievalQA.from_chain_type(
    llm=main_llm,
    retriever=ensemble_retriever,
    return_source_documents=True
)

# --- 7️⃣  Main flow ---
user_query = input("Ask a question: ")

# Retrieval via fused retrievers
result = qa.invoke({"query": user_query})

answer_en = result["result"]

answer = answer_en

print("\n🧠 Answer:\n", answer)
print("\n📄 Source Documents:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source", "unknown"))

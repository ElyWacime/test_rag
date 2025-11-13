
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
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

ensemble_retriever = EnsembleRetriever(
    retrievers=[base_retriever, multi_retriever],
    weights=[0.5, 0.5],
)

# --- 4️⃣  Query decomposition function ---
def decompose_query(query: str):
    """Use LLM to break a query into smaller sub-queries."""
    prompt = f"""
    Decompose the following query into smaller 3 sub-queries that together answer it completely.
    Query: "{query}"
    Return only one sub-query per line, no explanations.
    """
    response = main_llm.invoke(prompt)
    lines = [line.strip("-• ") for line in response.content.split("\n") if line.strip()]
    return lines

# --- 5️⃣  Decomposed retrieval + synthesis ---
def retrieve_with_decomposition(query: str):
    subqueries = decompose_query(query)
    print("\n🧩 Decomposed sub-queries:")
    for q in subqueries:
        print(" -", q)

    # Collect documents from each sub-query via RRF retriever
    all_docs = []
    for subq in subqueries:
        run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
        docs = ensemble_retriever._get_relevant_documents(subq, run_manager=run_manager)
        all_docs.extend(docs)

    # Deduplicate docs by content
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()

    # Synthesize final answer
    context = "\n\n".join([doc.page_content for doc in unique_docs])
    synthesis_prompt = f"""
    Based on the following context, answer the user's question in a clear, complete way.

    User question: {query}

    Context:
    {context}
    """
    synthesis_response = main_llm.invoke(synthesis_prompt)
    return synthesis_response.content, list(unique_docs)

# --- 6️⃣  Main flow ---
user_query = input("Ask a question: ")

answer, source_docs = retrieve_with_decomposition(user_query)

print("\n🧠 Final Answer:\n", answer)
print("\n📄 Source Documents:")
for doc in source_docs:
    print("-", doc.metadata.get("source", "unknown"))

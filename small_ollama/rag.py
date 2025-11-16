import os
import time
import requests
import json
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema import Document
from typing import List
import warnings
warnings.filterwarnings("ignore")

# Progress bar utility
class ProgressBar:
    @staticmethod
    def show_progress(description, current, total, bar_length=40):
        percent = float(current) / total
        arrow = '█' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(arrow))
        print(f'\r{description}: [{arrow + spaces}] {current}/{total} ({percent:.0%})', end='', flush=True)
        if current == total:
            print()

    @staticmethod
    def step(description):
        print(f"🔄 {description}...")

    @staticmethod
    def success(description):
        print(f"✅ {description}")

    @staticmethod
    def info(description):
        print(f"ℹ️  {description}")

    @staticmethod
    def warning(description):
        print(f"⚠️  {description}")

    @staticmethod
    def error(description):
        print(f"❌ {description}")

# Initialize Ollama client
class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.check_connection()
    
    def check_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                ProgressBar.success("Connected to Ollama")
                return True
            else:
                ProgressBar.warning("Ollama is not responding properly")
                return False
        except:
            ProgressBar.warning("Cannot connect to Ollama. Make sure it's running on localhost:11434")
            return False

ollama_client = OllamaClient()

# --- 1️⃣ Load and preprocess documents with progress ---
ProgressBar.step("Initializing embeddings model")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

if os.path.exists("faiss_index"):
    ProgressBar.step("Loading existing FAISS index")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    ProgressBar.success("FAISS index loaded")
else:
    ProgressBar.step("Building new FAISS index")
    
    ProgressBar.step("Scanning documents directory")
    loader = DirectoryLoader("docs", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    ProgressBar.success(f"Found {len(documents)} documents")
    
    ProgressBar.step("Splitting documents into chunks")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    ProgressBar.success(f"Created {len(docs)} text chunks")
    
    ProgressBar.step("Creating embeddings")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    ProgressBar.success("FAISS index built and saved")

# --- 2️⃣ Define LLMs using Ollama ---
ProgressBar.step("Initializing Ollama language models")
main_llm = Ollama(
    model="qwen2.5:0.5b",
    temperature=0.3,
    num_predict=1000
)
ProgressBar.success("Language models ready")

# --- 3️⃣ Build retrievers ---
ProgressBar.step("Building retrieval system")

class SimpleRetriever:
    def __init__(self, vectorstore, k=10):
        self.vectorstore = vectorstore
        self.k = k
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        try:
            # First try simple similarity search without scores
            docs = self.vectorstore.similarity_search(query, k=self.k)
            return docs
        except Exception as e:
            ProgressBar.warning(f"Retrieval error: {e}")
            return []

base_retriever = SimpleRetriever(vectorstore, k=10)
ProgressBar.success("Retrieval system built")

# --- 4️⃣ Improved Query decomposition function ---
def decompose_query(query: str):
    """Use LLM to break a query into smaller sub-queries."""
    ProgressBar.step("Decomposing query into sub-queries")
    
    # Much simpler and more direct prompt
    prompt_template = """
    Break this question into 2 simpler questions that would help answer it:
    "{query}"
    
    Return exactly 2 questions, one per line, without any numbers, bullets, or explanations.
    Make them simple and direct.
    """
    
    try:
        response = main_llm.invoke(prompt_template.format(query=query))
        response_text = str(response)
        
        # Clean up the response - remove any code blocks, numbers, etc.
        lines = response_text.split('\n')
        subqueries = []
        
        for line in lines:
            # Clean each line
            clean_line = line.strip()
            # Remove common prefixes
            for prefix in ['-', '•', '*', '1.', '2.', '3.', '```', 'sql']:
                if clean_line.startswith(prefix):
                    clean_line = clean_line[len(prefix):].strip()
            
            # Remove any remaining quotes
            clean_line = clean_line.strip('"\'').strip()
            
            # Only keep reasonable looking questions
            if (clean_line and 
                len(clean_line) > 5 and 
                len(clean_line) < 100 and
                '?' in clean_line and
                not clean_line.startswith('SELECT') and
                not '```' in clean_line):
                subqueries.append(clean_line)
        
        # If we didn't get good subqueries, use fallbacks
        if not subqueries:
            # Simple fallback subqueries based on the original query
            if 'black hole' in query.lower():
                subqueries = [
                    "What is a black hole?",
                    "How are black holes formed?"
                ]
            elif 'quantum' in query.lower():
                subqueries = [
                    "What is quantum mechanics?",
                    "What are quantum particles?"
                ]
            else:
                # Generic fallback - just use the original query
                subqueries = [query]
        
        ProgressBar.success(f"Query decomposed into {len(subqueries)} sub-queries")
        return subqueries[:2]  # Limit to 2 max
        
    except Exception as e:
        ProgressBar.warning(f"Query decomposition failed: {e}")
        return [query]  # Fallback to original query

# --- 5️⃣ Direct retrieval without complex decomposition ---
def retrieve_documents(query: str):
    ProgressBar.info(f"Processing query: '{query}'")
    
    # Try direct retrieval first (simpler and more reliable)
    ProgressBar.step("Retrieving relevant documents")
    
    try:
        # Get documents using the original query
        docs = base_retriever.get_relevant_documents(query)
        
        if docs:
            ProgressBar.success(f"Found {len(docs)} documents directly")
            
            # Simple deduplication
            seen_content = set()
            unique_docs = []
            for doc in docs:
                content_preview = doc.page_content[:100]  # Use first 100 chars for dedup
                if content_preview not in seen_content:
                    seen_content.add(content_preview)
                    unique_docs.append(doc)
            
            ProgressBar.info(f"After deduplication: {len(unique_docs)} documents")
            return unique_docs[:6]  # Return top 6 documents
            
        else:
            # If direct retrieval fails, try with decomposition
            ProgressBar.warning("No documents found with direct retrieval, trying decomposition")
            subqueries = decompose_query(query)
            
            all_docs = []
            for subq in subqueries:
                sub_docs = base_retriever.get_relevant_documents(subq)
                if sub_docs:
                    all_docs.extend(sub_docs)
                    ProgressBar.info(f"Found {len(sub_docs)} documents for: '{subq}'")
            
            if all_docs:
                # Deduplicate
                seen_content = set()
                unique_docs = []
                for doc in all_docs:
                    content_preview = doc.page_content[:100]
                    if content_preview not in seen_content:
                        seen_content.add(content_preview)
                        unique_docs.append(doc)
                
                ProgressBar.success(f"Found {len(unique_docs)} documents via decomposition")
                return unique_docs[:6]
            else:
                ProgressBar.error("No documents found with any method")
                return []
                
    except Exception as e:
        ProgressBar.error(f"Retrieval failed: {e}")
        return []

# --- 6️⃣ Answer generation ---
def generate_answer(query: str, documents: List[Document]):
    if not documents:
        return "I cannot answer this question based on the available documents. The information about this topic is not in my knowledge base.", []

    ProgressBar.step("Generating answer")
    
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Better prompt that encourages using the context
    synthesis_prompt = f"""
    Use the following context to answer the question. If the context contains relevant information, use it to answer. If not, say you cannot answer based on the available information.

    Context:
    {context}

    Question: {query}

    Answer based on the context above:
    """
    
    print("🤖 Generating answer...", end="", flush=True)
    try:
        synthesis_response = main_llm.invoke(synthesis_prompt)
        print(" Done!")
        
        answer_text = str(synthesis_response)
        ProgressBar.success("Answer generated successfully")
        return answer_text, documents
        
    except Exception as e:
        ProgressBar.error(f"Answer generation failed: {e}")
        return "I encountered an error while generating the answer. Please try again.", []

# --- 7️⃣ Main flow ---
def main():
    print("🤖 Local RAG System (100% Ollama)")
    print("📊 Enhanced with progress tracking")
    print("🔍 Simple and reliable retrieval")
    print("=" * 50)
    
    # Check available models
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"📚 Available Ollama models: {len(models)}")
            for model in models[:3]:
                print(f"   • {model['name']}")
    except:
        print("⚠️  Could not fetch model list")
    
    print("=" * 50)
    
    while True:
        user_query = input("\n🎯 Ask a question (or 'quit' to exit): ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not user_query:
            continue
            
        try:
            start_time = time.time()
            
            # Direct approach: retrieve then generate
            documents = retrieve_documents(user_query)
            answer, source_docs = generate_answer(user_query, documents)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print("\n" + "="*60)
            print("🧠 FINAL ANSWER:")
            print("="*60)
            print(answer)
            print("\n" + "="*60)
            
            if source_docs:
                print(f"\n📊 STATISTICS:")
                print(f"   • Processing time: {processing_time:.2f} seconds")
                print(f"   • Source documents used: {len(source_docs)}")
                
                print(f"\n📄 SOURCE DOCUMENTS:")
                for i, doc in enumerate(source_docs, 1):
                    source = doc.metadata.get("source", "unknown")
                    print(f"   {i}. {os.path.basename(source)}")
                    # Show a small preview
                    preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"      {preview}")
            else:
                print("\n📊 No source documents were used for this answer.")
                
            print("="*60)
                
        except Exception as e:
            ProgressBar.error(f"Error: {e}")
            print("💡 Please try a different question.")

if __name__ == "__main__":
    main()

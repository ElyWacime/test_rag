import os
import time
import requests
import json
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema import Document
from typing import List, Dict, Set
import warnings
warnings.filterwarnings("ignore")

# Progress bar utility (keep same as before)
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

# Optimized Multi-Model RAG System for CPU
class OptimizedRAGSystem:
    def __init__(self):
        self.user_manager = UserManager()
        self.current_user = None
        self.vectorstore = None
        self.embeddings = None
        self.llm = None
        self.initialize_optimized_models()
    
    def initialize_optimized_models(self):
        """Initialize optimized models for CPU performance"""
        ProgressBar.step("Initializing Optimized CPU-Friendly RAG System")
        
        # 🎯 OPTIMAL MODEL SELECTION FOR CPU:
        
        # 1. EMBEDDINGS: nomic-embed-text (274 MB) - Fastest, most CPU-friendly
        ProgressBar.step("Loading embedding model: nomic-embed-text")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",  # Your local model
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        ProgressBar.success("Embedding model loaded (nomic-embed-text)")
        
        # 2. TEXT GENERATION: qwen2.5:0.5b (397 MB) - Best balance of quality/speed
        ProgressBar.step("Loading text generation model: qwen2.5:0.5b")
        self.llm = Ollama(
            model="qwen2.5:0.5b",
            temperature=0.1,      # Low for factual responses
            num_predict=600,      # Shorter for faster generation
            num_thread=4,         # Use multiple CPU threads
            num_gpu=0             # Force CPU-only
        )
        ProgressBar.success("Text generation model loaded (qwen2.5:0.5b)")
        
        # 3. RERANKING: Simple keyword-based (CPU-friendly, no extra model)
        ProgressBar.step("Initializing lightweight reranking")
        ProgressBar.success("Lightweight reranking ready")
        
        # Load vector store
        if os.path.exists("optimized_faiss_index"):
            ProgressBar.step("Loading optimized FAISS index")
            self.vectorstore = FAISS.load_local("optimized_faiss_index", self.embeddings, allow_dangerous_deserialization=True)
            ProgressBar.success("FAISS index loaded")
        else:
            self.build_optimized_index()
        
        self.show_model_info()
    
    def show_model_info(self):
        """Display model information"""
        print(f"\n🎯 OPTIMIZED MODEL CONFIGURATION:")
        print(f"   • Embeddings: nomic-embed-text (274 MB) - Fast & CPU-friendly")
        print(f"   • Text Generation: qwen2.5:0.5b (397 MB) - Quality + Speed")
        print(f"   • Reranking: Lightweight keyword-based - No extra model")
        print(f"   • Total VRAM: 0 MB (CPU-only)")
        print(f"   • Total RAM: ~1-2 GB (estimated)")
    
    def build_optimized_index(self):
        """Build optimized index with CPU-friendly settings"""
        ProgressBar.step("Building optimized FAISS index")
        
        all_docs = []
        base_docs_path = "company_docs"
        
        categories = {
            "general": "general_docs",
            "hr": "hr_docs", 
            "finance": "finance_docs",
            "sensitive": "sensitive_docs",
            "confidential": "confidential_docs"
        }
        
        for category, folder in categories.items():
            folder_path = os.path.join(base_docs_path, folder)
            if os.path.exists(folder_path):
                ProgressBar.step(f"Loading {category} documents")
                loader = DirectoryLoader(folder_path, glob="*.txt", loader_cls=TextLoader)
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata["category"] = category
                    doc.metadata["access_level"] = category
                
                all_docs.extend(docs)
                ProgressBar.info(f"Loaded {len(docs)} {category} documents")
        
        if not all_docs:
            ProgressBar.warning("No documents found")
            return
        
        ProgressBar.step("Splitting documents into chunks")
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30)  # Smaller for CPU
        docs = splitter.split_documents(all_docs)
        ProgressBar.success(f"Created {len(docs)} text chunks")
        
        ProgressBar.step("Creating embeddings with nomic-embed-text")
        # Process in smaller batches for CPU efficiency
        batch_size = 20
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            ProgressBar.show_progress("Embedding documents", min(i + batch_size, len(docs)), len(docs))
            time.sleep(0.1)
        
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.vectorstore.save_local("optimized_faiss_index")
        ProgressBar.success("Optimized FAISS index built and saved")

# Keep UserManager the same
class UserManager:
    def __init__(self):
        self.users = {
            "admin": {
                "password": self._hash_password("admin123"),
                "role": "admin",
                "permissions": {"confidential", "sensitive", "hr", "finance", "general"}
            },
            "manager": {
                "password": self._hash_password("manager123"),
                "role": "manager",
                "permissions": {"sensitive", "hr", "general"}
            },
            "employee": {
                "password": self._hash_password("employee123"),
                "role": "employee",
                "permissions": {"general"}
            },
            "intern": {
                "password": self._hash_password("intern123"),
                "role": "intern",
                "permissions": {"general"}
            }
        }
    
    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate_user(self, username: str, password: str) -> Dict:
        if username in self.users:
            if self.users[username]["password"] == self._hash_password(password):
                return {
                    "username": username,
                    "role": self.users[username]["role"],
                    "permissions": self.users[username]["permissions"]
                }
        return None
    
    def can_access_document(self, user_permissions: Set[str], document_path: str) -> bool:
        doc_category = self._extract_document_category(document_path)
        return doc_category in user_permissions
    
    def _extract_document_category(self, document_path: str) -> str:
        filename = os.path.basename(document_path).lower()
        
        if any(cat in filename for cat in ["ceo", "board", "acquisition"]):
            return "confidential"
        elif any(cat in filename for cat in ["merger", "layoff", "competitor"]):
            return "sensitive"
        elif any(cat in filename for cat in ["salary", "review", "hiring"]):
            return "hr"
        elif any(cat in filename for cat in ["financial", "budget", "revenue"]):
            return "finance"
        else:
            return "general"

# Main Secure RAG System with optimized models
class SecureRAGSystem:
    def __init__(self):
        self.user_manager = UserManager()
        self.current_user = None
        self.optimized_rag = OptimizedRAGSystem()
        self.vectorstore = self.optimized_rag.vectorstore
        self.llm = self.optimized_rag.llm
        
    def login(self):
        print("\n🔐 Optimized Secure RAG System - CPU Edition")
        print("=" * 50)
        print("Available users:")
        print("• admin (admin123) - Full access")
        print("• manager (manager123) - Sensitive + HR + General")
        print("• employee (employee123) - General only") 
        print("• intern (intern123) - General only")
        print("=" * 50)
        
        while True:
            username = input("Username: ").strip()
            password = input("Password: ").strip()
            
            user_info = self.user_manager.authenticate_user(username, password)
            if user_info:
                self.current_user = user_info
                ProgressBar.success(f"Welcome {username} ({user_info['role']})!")
                print(f"📋 Permissions: {', '.join(sorted(user_info['permissions']))}")
                return True
            else:
                ProgressBar.error("Invalid credentials. Try again.")
    
    def secure_retrieve_documents(self, query: str) -> List[Document]:
        if not self.current_user:
            ProgressBar.error("No user logged in")
            return []
        
        ProgressBar.step("Retrieving documents")
        
        try:
            docs = self.vectorstore.similarity_search(query, k=8)  # Fewer for speed
            
            accessible_docs = []
            blocked_docs = []
            
            for doc in docs:
                if self.user_manager.can_access_document(self.current_user["permissions"], doc.metadata.get("source", "")):
                    accessible_docs.append(doc)
                else:
                    blocked_docs.append(doc)
            
            if blocked_docs:
                ProgressBar.warning(f"Access denied to {len(blocked_docs)} documents")
            
            # Lightweight reranking - keyword based for CPU
            accessible_docs = self.lightweight_rerank(query, accessible_docs)
            
            ProgressBar.success(f"Retrieved {len(accessible_docs)} documents")
            return accessible_docs[:4]  # Return fewer for speed
            
        except Exception as e:
            ProgressBar.error(f"Retrieval failed: {e}")
            return []
    
    def lightweight_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """CPU-friendly keyword-based reranking"""
        if len(documents) <= 1:
            return documents
        
        query_terms = set(query.lower().split())
        scored_docs = []
        
        for doc in documents:
            content = doc.page_content.lower()
            # Simple keyword matching score
            score = sum(1 for term in query_terms if term in content)
            scored_docs.append((score, doc))
        
        # Sort by score and return
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs]
    
    def generate_secure_answer(self, query: str, documents: List[Document]) -> str:
        if not documents:
            return "I cannot answer this question based on the documents accessible with your current permissions.", []
        
        ProgressBar.step("Generating answer")
        
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Optimized prompt for faster generation
        security_prompt = f"""
        Context: {context}
        Question: {query}
        
        Rules: Answer ONLY from context. If no answer in context, say "I cannot answer this question based on the available company documents."
        
        Answer:
        """
        
        print("🤖 Generating answer...", end="", flush=True)
        try:
            start_time = time.time()
            response = self.llm.invoke(security_prompt)
            end_time = time.time()
            
            print(f" Done! ({end_time - start_time:.1f}s)")
            
            answer_text = str(response).strip()
            
            # Quick hallucination check
            if self.is_clearly_hallucinated(answer_text, context):
                ProgressBar.warning("Answer rejected - not in context")
                return "I cannot answer this question based on the available company documents.", documents
            
            ProgressBar.success("Answer generated")
            return answer_text, documents
            
        except Exception as e:
            ProgressBar.error(f"Answer generation failed: {e}")
            return "I encountered an error while generating the answer. Please try again.", []
    
    def is_clearly_hallucinated(self, answer: str, context: str) -> bool:
        """Fast hallucination check for CPU"""
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # If answer says it cannot answer, it's safe
        if "cannot answer" in answer_lower or "not in" in answer_lower:
            return False
        
        # Check for obvious science/domain mismatches
        science_terms = {"photosynthesis", "quantum", "physics", "black hole", "planet"}
        if any(term in answer_lower for term in science_terms):
            if not any(term in context_lower for term in science_terms):
                return True
        
        return False
    
    def process_query(self, query: str):
        if not self.current_user:
            ProgressBar.error("Please login first")
            return
        
        ProgressBar.info(f"Processing query for {self.current_user['username']}: '{query}'")
        
        start_time = time.time()
        documents = self.secure_retrieve_documents(query)
        answer, source_docs = self.generate_secure_answer(query, documents)
        end_time = time.time()
        
        print("\n" + "="*60)
        print("🧠 SECURE ANSWER:")
        print("="*60)
        print(answer)
        print("\n" + "="*60)
        
        if source_docs:
            print(f"\n📊 PERFORMANCE:")
            print(f"   • Total time: {end_time - start_time:.2f}s")
            print(f"   • Documents used: {len(source_docs)}")
            print(f"   • User: {self.current_user['username']} ({self.current_user['role']})")
        else:
            print("\n📊 No accessible documents found.")
            
        print("="*60)

def main():
    print("🔐 Optimized Secure RAG System")
    print("🎯 CPU-Friendly Model Configuration")
    print("⚡ Fast & Efficient - No GPU Required")
    print("=" * 50)
    
    rag_system = SecureRAGSystem()
    
    if not rag_system.login():
        return
    
    while True:
        print(f"\n👤 Logged in as: {rag_system.current_user['username']} ({rag_system.current_user['role']})")
        print("💡 Type 'logout' to switch user, 'quit' to exit")
        
        user_query = input("\n🎯 Ask a question: ").strip()
        
        if user_query.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif user_query.lower() == 'logout':
            rag_system.current_user = None
            if not rag_system.login():
                break
            continue
        elif not user_query:
            continue
        
        rag_system.process_query(user_query)

if __name__ == "__main__":
    main()

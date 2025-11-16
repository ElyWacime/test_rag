import os
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import sqlite3
import json
from typing import List, Dict, Any

# Import our email modules
from email_downloader import GmailDownloader
from email_processor import EmailProcessor

load_dotenv()

class DebugOllamaEmailRAGSystem:
    def __init__(self, db_file="emails.db", chunks_db="email_chunks.db"):
        self.db_file = db_file
        self.chunks_db = chunks_db
        self.setup_models()
        self.setup_email_infrastructure()
    
    def setup_models(self):
        """Initialize all models using Ollama with timing"""
        print("🔄 Loading models with Ollama...")
        start_time = time.time()
        
        # 1. Embeddings: bge-m3 via Ollama
        print("📥 Loading BGE-M3 embeddings...", end=" ", flush=True)
        self.embeddings = OllamaEmbeddings(model="bge-m3")
        print(f"✅ ({time.time() - start_time:.1f}s)")
        
        # 2. LLM: llama3.2:3b via Ollama
        print("📥 Loading Llama 3.2 3B...", end=" ", flush=True)
        self.llm = Ollama(model="llama3.2:3b", temperature=0.3, num_predict=1000)
        print(f"✅ ({time.time() - start_time:.1f}s)")
        
        print(f"✅ All models loaded in {time.time() - start_time:.1f}s!")
    
    def setup_email_infrastructure(self):
        """Setup email downloader and processor"""
        print("📧 Setting up email infrastructure...", end=" ", flush=True)
        self.downloader = GmailDownloader(db_file=self.db_file)
        self.processor = EmailProcessor(db_file=self.db_file, chunks_db=self.chunks_db)
        print("✅")
    
    def check_existing_data(self):
        """Check if we already have email data"""
        if os.path.exists(self.db_file):
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM emails")
            email_count = cursor.fetchone()[0]
            conn.close()
            return email_count > 0
        return False
    
    def load_or_download_emails(self, max_emails=10, force_refresh=False):  # Reduced for testing
        """Load emails from database or download new ones"""
        print("📧 Email processing...", end=" ", flush=True)
        start_time = time.time()
        
        if force_refresh or not self.check_existing_data():
            print("\n   Downloading emails...", end=" ", flush=True)
            try:
                self.downloader.authenticate()
                self.downloader.get_all_emails(max_emails=max_emails)  # Small batch for testing
                print(f"✅ ({time.time() - start_time:.1f}s)")
            except Exception as e:
                print(f"❌ ({time.time() - start_time:.1f}s): {e}")
                return False
        else:
            print("Using existing database...", end=" ", flush=True)
        
        # Process emails for RAG
        print("\n   Processing emails for RAG...", end=" ", flush=True)
        processed_count = self.processor.process_emails_for_rag(batch_size=10)  # Small batch
        
        print(f"✅ Processed {processed_count} emails ({time.time() - start_time:.1f}s)")
        return True
    
    def load_email_documents(self):
        """Load email chunks as LangChain Documents"""
        print("📄 Loading email documents...", end=" ", flush=True)
        start_time = time.time()
        
        conn = sqlite3.connect(self.chunks_db)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM email_chunks')
        total_chunks = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT chunk_id, content, content_type, metadata 
            FROM email_chunks 
            ORDER BY email_id, chunk_index
        ''')
        
        documents = []
        for row in cursor.fetchall():
            chunk_id, content, content_type, metadata_json = row
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "gmail", "chunk_id": chunk_id, "content_type": content_type,
                        "email_id": metadata.get("email_id", ""), "subject": metadata.get("subject", ""),
                        "sender": metadata.get("sender", ""), "date": metadata.get("date", ""),
                        "chunk_type": metadata.get("chunk_type", "")
                    }
                )
                documents.append(doc)
            except json.JSONDecodeError:
                continue
        
        conn.close()
        print(f"✅ Loaded {len(documents)}/{total_chunks} chunks ({time.time() - start_time:.1f}s)")
        return documents
    
    def setup_vectorstore(self, force_rebuild=False):
        """Setup FAISS vectorstore with email documents"""
        print("🧠 Setting up vector store...")
        start_time = time.time()
        
        if os.path.exists("faiss_index") and not force_rebuild:
            print("   Loading existing FAISS index...", end=" ", flush=True)
            try:
                self.vectorstore = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
                print(f"✅ ({time.time() - start_time:.1f}s)")
                return self.vectorstore
            except Exception as e:
                print(f"❌ ({time.time() - start_time:.1f}s): {e}")
                print("   Rebuilding FAISS index...")
                force_rebuild = True
        
        if force_rebuild or not os.path.exists("faiss_index"):
            print("   Building new FAISS index...")
            
            # Load or download emails
            success = self.load_or_download_emails(max_emails=10)  # Small batch for testing
            if not success:
                raise Exception("Failed to load or download emails")
            
            # Load email documents
            email_documents = self.load_email_documents()
            
            if len(email_documents) == 0:
                raise Exception("No email documents found.")
            
            print(f"   Splitting {len(email_documents)} documents...", end=" ", flush=True)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(email_documents)
            print(f"✅ ({time.time() - start_time:.1f}s)")
            
            print(f"   Creating FAISS index with {len(docs)} documents...", end=" ", flush=True)
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            self.vectorstore.save_local("faiss_index")
            print(f"✅ ({time.time() - start_time:.1f}s)")
        
        print(f"✅ Vector store setup completed in {time.time() - start_time:.1f}s")
        return self.vectorstore
    
    def simple_query_test(self, query: str):
        """Simple test without complex decomposition"""
        print(f"\n🔍 Testing query: '{query}'")
        start_time = time.time()
        
        try:
            # Simple retrieval without decomposition
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            print("   Retrieving documents...", end=" ", flush=True)
            docs = retriever.get_relevant_documents(query)
            print(f"✅ Found {len(docs)} docs ({time.time() - start_time:.1f}s)")
            
            # Simple generation
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"Answer based on context: {query}\n\nContext:\n{context}\nAnswer:"
            
            print("   Generating answer...", end=" ", flush=True)
            answer = self.llm.invoke(prompt)
            print(f"✅ ({time.time() - start_time:.1f}s)")
            
            print(f"\n🧠 Answer: {answer}")
            return answer
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return None

def main():
    try:
        print("🚀 Debug RAG System - Starting with timing...")
        overall_start = time.time()
        
        # Initialize system
        rag_system = DebugOllamaEmailRAGSystem()
        
        # Setup vectorstore (this is likely where it hangs)
        vectorstore = rag_system.setup_vectorstore(force_rebuild=False)
        
        print(f"\n🎉 System ready in {time.time() - overall_start:.1f}s!")
        print("💡 Try a simple query...")
        
        # Test with a simple query
        test_queries = [
            "Show me recent emails",
            "What meetings do I have?",
            "Find emails about projects"
        ]
        
        for query in test_queries:
            rag_system.simple_query_test(query)
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ Main error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

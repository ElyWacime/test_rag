import os
import time
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import sqlite3
import json
from typing import List
from tqdm import tqdm  # For progress bars

load_dotenv()

class ProgressOllamaEmbeddings(OllamaEmbeddings):
    """Ollama embeddings with progress tracking"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with progress bar"""
        embeddings = []
        print("   Generating embeddings...")
        for i, text in enumerate(tqdm(texts, desc="      Embedding")):
            # Batch small groups to avoid overwhelming Ollama
            if i > 0 and i % 50 == 0:
                time.sleep(0.1)  # Small delay to prevent overload
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

class OptimizedEmailRAGSystem:
    def __init__(self, db_file="emails.db", chunks_db="email_chunks.db"):
        self.db_file = db_file
        self.chunks_db = chunks_db
        self.setup_models()
    
    def setup_models(self):
        """Initialize models"""
        print("🔄 Loading models...")
        self.embeddings = ProgressOllamaEmbeddings(model="bge-m3")
        self.llm = Ollama(model="llama3.2:3b", temperature=0.3, num_predict=500)
        print("✅ Models loaded")
    
    def load_email_documents(self):
        """Load email chunks as LangChain Documents"""
        print("📄 Loading email documents...")
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
        print(f"✅ Loaded {len(documents)} chunks")
        return documents
    
    def setup_vectorstore_fast(self, force_rebuild=False):
        """Optimized vectorstore setup with progress tracking"""
        print("🧠 Setting up vector store...")
        start_time = time.time()
        
        if os.path.exists("faiss_index") and not force_rebuild:
            print("   Loading existing FAISS index...")
            try:
                self.vectorstore = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
                print(f"✅ FAISS index loaded ({time.time() - start_time:.1f}s)")
                return self.vectorstore
            except Exception as e:
                print(f"❌ Error loading: {e}")
                print("   Rebuilding...")
        
        # Load documents
        documents = self.load_email_documents()
        
        if not documents:
            raise Exception("No documents found")
        
        print(f"   Processing {len(documents)} documents...")
        
        # Split documents (you already have chunks, so minimal splitting)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)  # Larger chunks
        docs = splitter.split_documents(documents)
        print(f"   After splitting: {len(docs)} documents")
        
        # Create FAISS index in batches to show progress
        print("   Creating FAISS index...")
        
        # Method 1: Use FAISS with progress (this is the standard approach)
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        
        print("   Saving index...")
        self.vectorstore.save_local("faiss_index")
        
        total_time = time.time() - start_time
        print(f"✅ FAISS index created with {len(docs)} documents ({total_time:.1f}s)")
        
        return self.vectorstore
    
    def setup_vectorstore_batch(self, force_rebuild=False, batch_size=100):
        """Alternative: Create FAISS index in batches"""
        print("🧠 Setting up vector store (batched)...")
        start_time = time.time()
        
        if os.path.exists("faiss_index") and not force_rebuild:
            print("   Loading existing FAISS index...")
            try:
                self.vectorstore = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
                print(f"✅ FAISS index loaded ({time.time() - start_time:.1f}s)")
                return self.vectorstore
            except Exception as e:
                print(f"❌ Error loading: {e}")
        
        # Load documents
        documents = self.load_email_documents()
        
        if not documents:
            raise Exception("No documents found")
        
        print(f"   Processing {len(documents)} documents in batches of {batch_size}...")
        
        # Process in batches
        all_embeddings = []
        all_docs = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_texts = [doc.page_content for doc in batch]
            
            print(f"      Batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")
            
            # Get embeddings for this batch
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            all_docs.extend(batch)
            
            # Show progress
            progress = min(i + batch_size, len(documents))
            print(f"      Progress: {progress}/{len(documents)} documents embedded")
        
        # Create FAISS index from all embeddings
        print("   Creating FAISS index from embeddings...")
        embeddings_array = np.array(all_embeddings)
        
        # Use the FAISS internal method to create index
        from langchain_community.vectorstores.utils import maximal_marginal_relevance
        import faiss
        
        # Create FAISS index
        dimension = len(all_embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add embeddings to index
        index.add(embeddings_array.astype('float32'))
        
        # Create FAISS vector store
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=FAISS._build_docstore(all_docs),
            index_to_docstore_id=FAISS._build_index_to_docstore_id(all_docs)
        )
        
        print("   Saving index...")
        self.vectorstore.save_local("faiss_index")
        
        total_time = time.time() - start_time
        print(f"✅ FAISS index created with {len(all_docs)} documents ({total_time:.1f}s)")
        
        return self.vectorstore
    
    def quick_test(self):
        """Quick test to verify everything works"""
        print("\n🔍 Quick test...")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        
        test_queries = [
            "recent emails",
            "meetings",
            "projects"
        ]
        
        for query in test_queries:
            print(f"   Query: '{query}'")
            try:
                docs = retriever.get_relevant_documents(query)
                print(f"      Found {len(docs)} documents")
            except Exception as e:
                print(f"      Error: {e}")

def main():
    try:
        print("🚀 Optimized RAG System")
        print("=" * 50)
        
        # Install tqdm if not available
        try:
            from tqdm import tqdm
        except ImportError:
            print("Installing tqdm for progress bars...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
            from tqdm import tqdm
        
        rag_system = OptimizedEmailRAGSystem()
        
        # Try the fast method first
        print("\n1. Trying standard method...")
        try:
            vectorstore = rag_system.setup_vectorstore_fast(force_rebuild=True)
            print("✅ Standard method succeeded!")
        except Exception as e:
            print(f"❌ Standard method failed: {e}")
            print("\n2. Trying batched method...")
            vectorstore = rag_system.setup_vectorstore_batch(force_rebuild=True, batch_size=50)
        
        # Quick test
        rag_system.quick_test()
        
        print("\n🎉 System ready!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

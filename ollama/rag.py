import os
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

class QwenReranker:
    def __init__(self, model_name="dengcao/Qwen3-Reranker-0.6B:Q8_0"):
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=0)
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """Rerank documents using Qwen3 reranker model"""
        if not documents or len(documents) <= 1:
            return documents[:top_k]
            
        print(f"🔍 Reranking {len(documents)} documents with Qwen3 reranker...")
        
        try:
            scores = []
            for i, doc in enumerate(documents):
                # Qwen3 reranker expects a specific format
                # Based on the model card, it likely expects query-document pairs
                prompt = self._create_reranker_prompt(query, doc.page_content)
                
                # Get relevance score from the reranker model
                response = self.llm.invoke(prompt)
                
                # Extract score from response
                score = self._extract_score(response)
                scores.append(score)
                
                # Show progress for large batches
                if (i + 1) % 5 == 0:
                    print(f"   Processed {i + 1}/{len(documents)} documents...")
            
            # Sort documents by score (higher is better)
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            print(f"✅ Reranking completed. Top score: {scored_docs[0][0] if scored_docs else 'N/A'}")
            return [doc for score, doc in scored_docs[:top_k]]
            
        except Exception as e:
            print(f"⚠️  Reranking failed, using original order: {e}")
            return documents[:top_k]
    
    def _create_reranker_prompt(self, query: str, document: str) -> str:
        """Create prompt for Qwen3 reranker"""
        # Limit document length to avoid context issues
        document_preview = document[:1000]  # First 1000 characters
        
        # Try different prompt formats that might work with this model
        prompt_formats = [
            # Format 1: Direct scoring request
            f"""Score the relevance between this query and document on a scale of 0.0 to 1.0.

Query: {query}

Document: {document_preview}

Relevance score: """,
            
            # Format 2: Instruction-based
            f"""Evaluate how relevant this document is to the query. Return only a number between 0.0 and 1.0.

Query: {query}
Document: {document_preview}

Score: """,
            
            # Format 3: Simple format
            f"""Query: {query}
Document: {document_preview}
Relevance Score: """
        ]
        
        return prompt_formats[0]  # Start with the first format
    
    def _extract_score(self, response: str) -> float:
        """Extract numerical score from model response"""
        try:
            # Clean the response
            clean_response = response.strip()
            
            # Try to find numerical values in different formats
            # Look for floats like 0.85, 0.9, etc.
            score_patterns = [
                r"[-+]?\d*\.\d+",  # Float numbers
                r"[-+]?\d+",       # Integer numbers
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, clean_response)
                if matches:
                    # Take the first number found
                    score = float(matches[0])
                    # Normalize to 0-1 range if needed
                    if score > 1.0:
                        score = score / 100.0  # Assume percentage if > 1
                    elif score < 0.0:
                        score = 0.0
                    return min(max(score, 0.0), 1.0)  # Clamp between 0-1
            
            # Fallback: if no number found, try to interpret text
            if any(word in clean_response.lower() for word in ["high", "relevant", "good", "excellent"]):
                return 0.8
            elif any(word in clean_response.lower() for word in ["medium", "moderate", "average"]):
                return 0.5
            elif any(word in clean_response.lower() for word in ["low", "irrelevant", "poor"]):
                return 0.2
            else:
                return 0.5  # Default moderate score
                
        except (ValueError, IndexError):
            print(f"⚠️  Could not parse score from response: {response}")
            return 0.5  # Default score

class OllamaEmailRAGSystem:
    def __init__(self, db_file="emails.db", chunks_db="email_chunks.db"):
        self.db_file = db_file
        self.chunks_db = chunks_db
        self.setup_models()
        self.setup_email_infrastructure()
    
    def setup_models(self):
        """Initialize all models using Ollama"""
        print("🔄 Loading models with Ollama...")
        
        # 1. Embeddings: bge-m3 via Ollama
        print("📥 Loading BGE-M3 embeddings...")
        self.embeddings = OllamaEmbeddings(
            model="bge-m3",
            base_url="http://localhost:11434"
        )
        
        # 2. LLM: llama3.2:3b via Ollama
        print("📥 Loading Llama 3.2 3B...")
        self.llm = Ollama(
            model="llama3.2:3b",
            base_url="http://localhost:11434",
            temperature=0.3,
            num_predict=1000
        )
        
        # 3. Reranker: Qwen3-Reranker via Ollama
        print("📥 Loading Qwen3-Reranker-0.6B...")
        self.reranker = QwenReranker("dengcao/Qwen3-Reranker-0.6B:Q8_0")
        
        print("✅ All models loaded successfully!")
    
    def setup_email_infrastructure(self):
        """Setup email downloader and processor"""
        self.downloader = GmailDownloader(db_file=self.db_file)
        self.processor = EmailProcessor(db_file=self.db_file, chunks_db=self.chunks_db)
    
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
    
    def load_or_download_emails(self, max_emails=100, force_refresh=False):
        """Load emails from database or download new ones"""
        if force_refresh or not self.check_existing_data():
            print("📧 Downloading emails from Gmail...")
            try:
                self.downloader.authenticate()
                self.downloader.get_all_emails(max_emails=max_emails)
                print("✅ Email download completed")
            except Exception as e:
                print(f"❌ Error downloading emails: {e}")
                return False
        else:
            print("📂 Using existing email database")
        
        # Process emails for RAG
        print("🔄 Processing emails for RAG...")
        processed_count = self.processor.process_emails_for_rag(batch_size=100)
        
        if processed_count == 0:
            print("⚠️  No new emails to process.")
        
        return True
    
    def load_email_documents(self):
        """Load email chunks as LangChain Documents"""
        conn = sqlite3.connect(self.chunks_db)
        cursor = conn.cursor()
        
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
                
                # Create LangChain Document
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "gmail",
                        "chunk_id": chunk_id,
                        "content_type": content_type,
                        "email_id": metadata.get("email_id", ""),
                        "subject": metadata.get("subject", ""),
                        "sender": metadata.get("sender", ""),
                        "date": metadata.get("date", ""),
                        "chunk_type": metadata.get("chunk_type", "")
                    }
                )
                documents.append(doc)
            except json.JSONDecodeError:
                print(f"⚠️  Could not parse metadata for chunk {chunk_id}")
                continue
        
        conn.close()
        print(f"📄 Loaded {len(documents)} email chunks")
        return documents
    
    def setup_vectorstore(self, force_rebuild=False):
        """Setup FAISS vectorstore with email documents"""
        if os.path.exists("faiss_index") and not force_rebuild:
            print("📂 Loading existing FAISS index...")
            try:
                self.vectorstore = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
                print("✅ FAISS index loaded successfully")
                return self.vectorstore
            except Exception as e:
                print(f"❌ Error loading FAISS index: {e}")
                print("🔄 Rebuilding FAISS index...")
                force_rebuild = True
        
        if force_rebuild or not os.path.exists("faiss_index"):
            print("🧠 Building new FAISS index from emails...")
            
            # Load or download emails
            success = self.load_or_download_emails(max_emails=100)
            if not success:
                raise Exception("Failed to load or download emails")
            
            # Load email documents
            email_documents = self.load_email_documents()
            
            if len(email_documents) == 0:
                raise Exception("No email documents found to build vector store.")
            
            print(f"📝 Processing {len(email_documents)} email chunks...")
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(email_documents)
            
            if len(docs) == 0:
                raise Exception("No documents after splitting. Check your email content.")
            
            print(f"🔨 Creating FAISS index with {len(docs)} documents...")
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            self.vectorstore.save_local("faiss_index")
            print("✅ FAISS index created and saved")
        
        return self.vectorstore
    
    def create_retrievers(self, vectorstore):
        """Create ensemble retriever"""
        # Semantic retriever
        semantic_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}
        )
        
        # BM25 retriever (keyword-based)
        email_documents = self.load_email_documents()
        bm25_retriever = BM25Retriever.from_documents(email_documents)
        bm25_retriever.k = 20
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        return ensemble_retriever
    
    def decompose_query(self, query: str):
        """Use Ollama LLM to break query into sub-queries"""
        prompt = f"""You are a helpful assistant that decomposes complex questions into simpler sub-questions.
Break the following query into 2-3 smaller sub-queries that together answer it completely.

Query: "{query}"

Return only the sub-queries, one per line, without any explanations or numbering."""

        try:
            response = self.llm.invoke(prompt)
            lines = [line.strip("-• ") for line in response.split("\n") if line.strip()]
            return lines[:3]
        except Exception as e:
            print(f"⚠️  Query decomposition failed: {e}")
            return [query]
    
    def retrieve_with_reranking(self, query: str, ensemble_retriever, top_k: int = 8):
        """Retrieve documents with decomposition and Qwen3 reranking"""
        # Step 1: Query decomposition
        subqueries = self.decompose_query(query)
        print("\n🧩 Decomposed sub-queries:")
        for i, q in enumerate(subqueries, 1):
            print(f" {i}. {q}")

        # Step 2: Retrieve documents for each sub-query
        all_docs = []
        for subq in subqueries:
            try:
                docs = ensemble_retriever.get_relevant_documents(subq)
                all_docs.extend(docs)
            except Exception as e:
                print(f"⚠️  Retrieval failed for sub-query: {e}")
                continue
        
        # Step 3: Deduplicate
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        unique_docs_list = list(unique_docs)
        
        if len(unique_docs_list) > 1:
            # Step 4: Rerank with Qwen3 reranker
            reranked_docs = self.reranker.rerank(query, unique_docs_list, top_k=top_k)
            return reranked_docs
        else:
            return unique_docs_list[:top_k]
    
    def generate_answer(self, query: str, context_docs: List[Document]):
        """Generate answer using Ollama LLM"""
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""You are a helpful email assistant. Answer the user's question based ONLY on the provided email context.
Be concise and factual. Cite relevant senders and dates when possible.
If the context doesn't contain the answer, say so clearly.

Question: {query}

Email Context:
{context}

Answer:"""

        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Error generating answer: {e}"

def main():
    try:
        # Initialize Ollama RAG system
        print("🚀 Initializing Ollama Email RAG System...")
        print("📋 Models being used:")
        print("  - Embeddings: bge-m3")
        print("  - LLM: llama3.2:3b") 
        print("  - Reranker: Qwen3-Reranker-0.6B")
        print("\n⏳ Please ensure Ollama is running: ollama serve")
        
        rag_system = OllamaEmailRAGSystem()
        
        # Setup vectorstore
        print("🔄 Setting up vector store...")
        vectorstore = rag_system.setup_vectorstore(force_rebuild=False)
        
        # Create retrievers
        ensemble_retriever = rag_system.create_retrievers(vectorstore)
        
        print("\n✅ Ollama RAG System Ready!")
        print("🔍 You can now ask questions about your emails!")
        
        while True:
            user_query = input("\nAsk a question about your emails (or 'quit' to exit): ")
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            print("\n🤔 Processing your query...")
            
            # Retrieve with decomposition and reranking
            source_docs = rag_system.retrieve_with_reranking(user_query, ensemble_retriever)
            
            # Generate answer
            answer = rag_system.generate_answer(user_query, source_docs)
            
            print("\n🧠 Final Answer:\n", answer)
            print(f"\n📧 Top Source Emails ({len(source_docs)}):")
            for i, doc in enumerate(source_docs[:5], 1):
                metadata = doc.metadata
                print(f"{i}. From: {metadata.get('sender', 'Unknown')}")
                print(f"   Subject: {metadata.get('subject', 'No subject')}")
                print(f"   Date: {metadata.get('date', 'Unknown')}")
                print()
                
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

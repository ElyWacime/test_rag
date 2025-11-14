import os
import sqlite3
import json
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from dotenv import load_dotenv

# Import our email modules
from email_downloader import GmailDownloader
from email_processor import EmailProcessor

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

os.environ["OPENAI_API_KEY"] = LLM_API_KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

class EmailRAGSystem:
    def __init__(self, db_file="emails.db", chunks_db="email_chunks.db"):
        self.db_file = db_file
        self.chunks_db = chunks_db
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.setup_email_infrastructure()
    
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
            print("⚠️  No new emails to process. Checking if we need to download emails...")
            # If no emails were processed, try to download a small batch
            if not self.check_existing_data():
                print("🔄 No emails found. Attempting to download a small batch...")
                self.load_or_download_emails(max_emails=50, force_refresh=True)
        
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
                raise Exception("No email documents found to build vector store. Please check if emails were downloaded properly.")
            
            print(f"📝 Processing {len(email_documents)} email chunks...")
            
            # Split documents (optional, since we already chunked emails)
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

# --- Your existing RAG functions ---

def main():
    try:
        # Initialize email RAG system
        print("🚀 Initializing Email RAG System...")
        email_rag = EmailRAGSystem()
        
        # Setup vectorstore with emails
        print("🔄 Setting up vector store...")
        vectorstore = email_rag.setup_vectorstore(force_rebuild=False)
        
        # --- Your existing RAG code (unchanged) ---
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        main_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3)
        
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=main_llm,
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[base_retriever, multi_retriever],
            weights=[0.5, 0.5],
        )
        
        def decompose_query(query: str):
            prompt = f"""
            Decompose the following query into smaller 3 sub-queries that together answer it completely.
            Query: "{query}"
            Return only one sub-query per line, no explanations.
            """
            response = main_llm.invoke(prompt)
            lines = [line.strip("-• ") for line in response.content.split("\n") if line.strip()]
            return lines
        
        def retrieve_with_decomposition(query: str):
            subqueries = decompose_query(query)
            print("\n🧩 Decomposed sub-queries:")
            for q in subqueries:
                print(" -", q)

            all_docs = []
            for subq in subqueries:
                run_manager = CallbackManagerForRetrieverRun.get_noop_manager()
                docs = ensemble_retriever._get_relevant_documents(subq, run_manager=run_manager)
                all_docs.extend(docs)

            unique_docs = {doc.page_content: doc for doc in all_docs}.values()
            
            context = "\n\n".join([doc.page_content for doc in unique_docs])
            synthesis_prompt = f"""
            Based on the following context from the user's emails, answer the user's question in a clear, complete way.
            Focus on information from the emails and cite relevant senders and dates when possible.

            User question: {query}

            Context from emails:
            {context}
            """
            synthesis_response = main_llm.invoke(synthesis_prompt)
            return synthesis_response.content, list(unique_docs)
        
        # --- Main flow ---
        print("\n✅ Email RAG System Ready!")
        print("🔍 You can now ask questions about your emails!")
        
        while True:
            user_query = input("\nAsk a question about your emails (or 'quit' to exit): ")
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            answer, source_docs = retrieve_with_decomposition(user_query)
            
            print("\n🧠 Final Answer:\n", answer)
            print(f"\n📧 Source Emails ({len(source_docs)} found):")
            for doc in source_docs[:5]:  # Show top 5 sources
                metadata = doc.metadata
                print(f"- From: {metadata.get('sender', 'Unknown')}")
                print(f"  Subject: {metadata.get('subject', 'No subject')}")
                print(f"  Date: {metadata.get('date', 'Unknown')}")
                print(f"  Type: {metadata.get('chunk_type', 'content')}")
                print()
                
    except Exception as e:
        print(f"❌ Error in main: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if you have credentials.json in the same directory")
        print("2. Make sure you have internet connection for Gmail API")
        print("3. Verify your Google Cloud project has Gmail API enabled")
        print("4. Try running with force_rebuild=True")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

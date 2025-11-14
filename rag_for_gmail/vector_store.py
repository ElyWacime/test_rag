import sqlite3
import json
import numpy as np
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

class EmailVectorStore:
    def __init__(self, chunks_db='email_chunks.db', persist_directory="./chroma_db"):
        self.chunks_db = chunks_db
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="emails",
            metadata={"hnsw:space": "cosine"}
        )
    
    def embed_and_store_chunks(self, batch_size: int = 100):
        """Embed chunks and store in vector database"""
        conn = sqlite3.connect(self.chunks_db)
        cursor = conn.cursor()
        
        # Get chunks that haven't been vectorized yet
        cursor.execute('''
            SELECT c.* FROM email_chunks c
            LEFT JOIN vector_status v ON c.chunk_id = v.chunk_id
            WHERE v.chunk_id IS NULL
            LIMIT ?
        ''', (batch_size,))
        
        chunks = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        
        if not chunks:
            print("No new chunks to process.")
            return 0
        
        processed_chunks = []
        
        for chunk_row in chunks:
            chunk = dict(zip(column_names, chunk_row))
            
            # For production, you'd use OpenAI, Cohere, or local embeddings
            # This is a simplified version - you need to add proper embeddings
            try:
                # Store in ChromaDB
                self.collection.add(
                    documents=[chunk['content']],
                    metadatas=[{
                        'chunk_id': chunk['chunk_id'],
                        'email_id': chunk['email_id'],
                        'content_type': chunk['content_type'],
                        'chunk_index': chunk['chunk_index'],
                        'token_count': chunk['token_count'],
                        'original_metadata': chunk['metadata']
                    }],
                    ids=[chunk['chunk_id']]
                )
                
                processed_chunks.append(chunk['chunk_id'])
                
            except Exception as e:
                print(f"Error processing chunk {chunk['chunk_id']}: {e}")
                continue
        
        # Mark chunks as processed
        if processed_chunks:
            placeholders = ','.join('?' * len(processed_chunks))
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS vector_status (
                    chunk_id TEXT PRIMARY KEY,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.executemany(
                "INSERT OR REPLACE INTO vector_status (chunk_id) VALUES (?)",
                [(chunk_id,) for chunk_id in processed_chunks]
            )
            
            conn.commit()
        
        conn.close()
        
        print(f"Processed {len(processed_chunks)} chunks into vector store.")
        return len(processed_chunks)
    
    def search_emails(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search emails using semantic search"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []

import sqlite3
import re
import json
from typing import List, Dict, Any

class EmailProcessor:
    def __init__(self, db_file='emails.db', chunks_db='email_chunks.db'):
        self.db_file = db_file
        self.chunks_db = chunks_db
        self.setup_chunks_database()
    
    def setup_chunks_database(self):
        """Setup database for email chunks"""
        conn = sqlite3.connect(self.chunks_db)
        cursor = conn.cursor()
        
        # Create email_chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_chunks (
                chunk_id TEXT PRIMARY KEY,
                email_id TEXT,
                chunk_index INTEGER,
                content TEXT,
                content_type TEXT,
                token_count INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (email_id) REFERENCES emails (id)
            )
        ''')
        
        # Create indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_email_id ON email_chunks (email_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_content_type ON email_chunks (content_type)
        ''')
        
        conn.commit()
        conn.close()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove email headers and footers
        text = re.sub(r'On\s+.+wrote:', '', text)
        text = re.sub(r'From:.+\n', '', text)
        text = re.sub(r'Sent:.+\n', '', text)
        text = re.sub(r'To:.+\n', '', text)
        text = re.sub(r'Subject:.+\n', '', text)
        
        return text.strip()
    
    def process_emails_for_rag(self, batch_size: int = 100):
        """Process all unprocessed emails for RAG"""
        conn = sqlite3.connect(self.db_file)
        chunks_conn = sqlite3.connect(self.chunks_db)
        
        cursor = conn.cursor()
        chunks_cursor = chunks_conn.cursor()
        
        # Get unprocessed emails
        cursor.execute("SELECT * FROM emails WHERE processed = FALSE LIMIT ?", (batch_size,))
        emails = cursor.fetchall()
        
        column_names = [description[0] for description in cursor.description]
        processed_count = 0
        
        for email_row in emails:
            email = dict(zip(column_names, email_row))
            
            try:
                # Create chunks for different parts of the email
                chunks = self.create_email_chunks(email)
                
                # Store chunks
                for chunk in chunks:
                    chunks_cursor.execute('''
                        INSERT OR REPLACE INTO email_chunks 
                        (chunk_id, email_id, chunk_index, content, content_type, token_count, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        chunk['chunk_id'],
                        chunk['email_id'],
                        chunk['chunk_index'],
                        chunk['content'],
                        chunk['content_type'],
                        chunk['token_count'],
                        chunk['metadata']
                    ))
                
                # Mark email as processed
                cursor.execute("UPDATE emails SET processed = TRUE WHERE id = ?", (email['id'],))
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} emails...")
                
            except Exception as e:
                print(f"Error processing email {email['id']}: {e}")
                continue
        
        conn.commit()
        chunks_conn.commit()
        
        conn.close()
        chunks_conn.close()
        
        print(f"Processing complete! Processed {processed_count} emails.")
        return processed_count
    
    def create_email_chunks(self, email: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from email for RAG"""
        chunks = []
        
        # Clean and prepare content
        subject = self.clean_text(email['subject'])
        body_text = self.clean_text(email['body_text'])
        sender = self.clean_text(email['sender'])
        date = self.clean_text(email['date'])
        
        # Chunk 1: Subject and metadata (always include)
        metadata_content = f"From: {sender}\nDate: {date}\nSubject: {subject}\n\nSummary: {email['snippet']}"
        metadata_chunk = {
            'chunk_id': f"{email['id']}_meta_0",
            'email_id': email['id'],
            'chunk_index': 0,
            'content': metadata_content,
            'content_type': 'metadata',
            'token_count': len(metadata_content.split()),
            'metadata': json.dumps({
                'subject': subject,
                'sender': sender,
                'date': date,
                'chunk_type': 'metadata'
            })
        }
        chunks.append(metadata_chunk)
        
        # Chunk body text if available
        if body_text:
            # Simple chunking by sentences
            sentences = re.split(r'[.!?]+', body_text)
            current_chunk = []
            chunk_index = 1
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                current_chunk.append(sentence)
                
                # Create chunk when we have enough content
                if len(' '.join(current_chunk)) > 400:
                    chunk_content = f"From: {sender}\nDate: {date}\nSubject: {subject}\n\nContent: {' '.join(current_chunk)}"
                    
                    chunk = {
                        'chunk_id': f"{email['id']}_body_{chunk_index}",
                        'email_id': email['id'],
                        'chunk_index': chunk_index,
                        'content': chunk_content,
                        'content_type': 'body',
                        'token_count': len(chunk_content.split()),
                        'metadata': json.dumps({
                            'subject': subject,
                            'sender': sender,
                            'date': date,
                            'chunk_type': 'body',
                            'chunk_index': chunk_index
                        })
                    }
                    chunks.append(chunk)
                    
                    current_chunk = []
                    chunk_index += 1
            
            # Add remaining content
            if current_chunk:
                chunk_content = f"From: {sender}\nDate: {date}\nSubject: {subject}\n\nContent: {' '.join(current_chunk)}"
                chunk = {
                    'chunk_id': f"{email['id']}_body_{chunk_index}",
                    'email_id': email['id'],
                    'chunk_index': chunk_index,
                    'content': chunk_content,
                    'content_type': 'body',
                    'token_count': len(chunk_content.split()),
                    'metadata': json.dumps({
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'chunk_type': 'body',
                        'chunk_index': chunk_index
                    })
                }
                chunks.append(chunk)
        
        return chunks
    
    def get_chunk_count(self):
        """Get count of chunks in database"""
        conn = sqlite3.connect(self.chunks_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM email_chunks")
        count = cursor.fetchone()[0]
        conn.close()
        return count

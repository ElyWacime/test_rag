import os
import json
import base64
import pickle
import sqlite3
from datetime import datetime
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time

# Set the environment variable to allow HTTP for local development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

class GmailDownloader:
    def __init__(self, credentials_file='credentials.json', token_file='token.pickle', db_file='emails.db'):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.db_file = db_file
        self.service = None
        self.setup_database()
        
    def setup_database(self):
        """Setup SQLite database to store email metadata"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emails (
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                subject TEXT,
                sender TEXT,
                recipients TEXT,
                date TEXT,
                body_text TEXT,
                body_html TEXT,
                labels TEXT,
                snippet TEXT,
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def authenticate(self):
        """Use the exact Flask authentication method"""
        return self.authenticate_with_flow()
    
    def authenticate_with_flow(self):
        """Authenticate using Flow (like Flask does)"""
        creds = None
        
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Use Flow instead of InstalledAppFlow (like Flask)
                flow = Flow.from_client_secrets_file(
                    self.credentials_file,
                    scopes=self.SCOPES,
                    redirect_uri='http://localhost:8080/oauth2callback'
                )
                
                authorization_url, state = flow.authorization_url(
                    access_type='offline',
                    include_granted_scopes='true',
                    prompt='consent'
                )
                
                print("🔐 Please visit this URL to authorize the application:")
                print(authorization_url)
                print("\nAfter authorizing, you will be redirected to a localhost URL.")
                print("The page might show 'Connection refused' - this is normal.")
                print("Just copy the COMPLETE URL from your browser and paste it below.\n")
                
                authorization_response = input("Paste the complete redirect URL here: ").strip()
                
                # Fetch tokens
                flow.fetch_token(authorization_response=authorization_response)
                creds = flow.credentials
            
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
        return self.service
    
    def get_all_emails(self, batch_size=100, max_emails=None):
        """Get all emails with pagination"""
        if not self.service:
            self.authenticate()
        
        messages = []
        page_token = None
        total_processed = 0
        
        print("Fetching email list...")
        
        while True:
            try:
                result = self.service.users().messages().list(
                    userId='me',
                    pageToken=page_token,
                    maxResults=batch_size
                ).execute()
                
                if 'messages' in result:
                    batch_messages = result['messages']
                    messages.extend(batch_messages)
                    total_processed += len(batch_messages)
                    
                    print(f"Retrieved {len(batch_messages)} messages. Total: {total_processed}")
                    
                    # Process this batch
                    self.process_message_batch(batch_messages)
                
                page_token = result.get('nextPageToken')
                if not page_token or (max_emails and total_processed >= max_emails):
                    break
                    
                # Rate limiting
                time.sleep(0.1)
                
            except HttpError as error:
                print(f"Error fetching messages: {error}")
                break
        
        return messages
    
    def process_message_batch(self, messages):
        """Process a batch of messages and store in database"""
        conn = sqlite3.connect(self.db_file)
        
        for i, msg in enumerate(messages):
            try:
                # Check if already processed
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM emails WHERE id = ?", (msg['id'],))
                if cursor.fetchone():
                    continue
                
                # Get full message
                message = self.service.users().messages().get(
                    userId='me', 
                    id=msg['id'], 
                    format='full'
                ).execute()
                
                # Parse message
                email_data = self.parse_email_message(message)
                
                # Store in database
                cursor.execute('''
                    INSERT OR REPLACE INTO emails 
                    (id, thread_id, subject, sender, recipients, date, body_text, body_html, labels, snippet)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    email_data['id'],
                    email_data['thread_id'],
                    email_data['subject'],
                    email_data['sender'],
                    email_data['recipients'],
                    email_data['date'],
                    email_data['body_text'],
                    email_data['body_html'],
                    email_data['labels'],
                    email_data['snippet']
                ))
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(messages)} in current batch")
                
                # Rate limiting
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Error processing message {msg['id']}: {e}")
                continue
        
        conn.commit()
        conn.close()
    
    def parse_email_message(self, message):
        """Parse Gmail message into structured data"""
        # Extract headers
        headers = {h['name']: h['value'] for h in message['payload'].get('headers', [])}
        
        # Extract body content
        body_text, body_html = self.extract_body(message['payload'])
        
        return {
            'id': message['id'],
            'thread_id': message.get('threadId'),
            'subject': headers.get('Subject', ''),
            'sender': headers.get('From', ''),
            'recipients': headers.get('To', ''),
            'date': headers.get('Date', ''),
            'body_text': body_text,
            'body_html': body_html,
            'labels': ','.join(message.get('labelIds', [])),
            'snippet': message.get('snippet', '')
        }
    
    def extract_body(self, payload):
        """Extract text and HTML body from message payload"""
        body_text = ""
        body_html = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    body_text = self.decode_body(part['body'].get('data', ''))
                elif part['mimeType'] == 'text/html':
                    body_html = self.decode_body(part['body'].get('data', ''))
        else:
            if payload['mimeType'] == 'text/plain':
                body_text = self.decode_body(payload['body'].get('data', ''))
            elif payload['mimeType'] == 'text/html':
                body_html = self.decode_body(payload['body'].get('data', ''))
        
        return body_text, body_html
    
    def decode_body(self, data):
        """Decode base64 email body"""
        if data:
            try:
                return base64.urlsafe_b64decode(data).decode('utf-8')
            except:
                return ""
        return ""
    
    def get_email_count(self):
        """Get count of emails in database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM emails")
        count = cursor.fetchone()[0]
        conn.close()
        return count

def main():
    downloader = GmailDownloader()
    
    print("Authenticating...")
    downloader.authenticate()
    
    print("Starting email download...")
    
    # Download all emails (remove max_emails limit for full download)
    downloader.get_all_emails(max_emails=50)  # Start with 50 for testing
    
    count = downloader.get_email_count()
    print(f"Download complete! Total emails in database: {count}")

if __name__ == '__main__':
    main()

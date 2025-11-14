import os
import pickle
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# Set the environment variable to allow HTTP for local development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the callback URL
        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)
        
        # Send success response
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"""
        <html>
            <body>
                <h1>Authentication Successful!</h1>
                <p>You can close this window and return to the application.</p>
                <script>window.close();</script>
            </body>
        </html>
        """)
        
        # Store the authorization response for the main thread
        self.server.authorization_response = self.requestline.split(' ')[1]

    def log_message(self, format, *args):
        return  # Suppress log messages

def authenticate_gmail_exact_flask_style():
    """Replicate the exact Flask authentication flow"""
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    CLIENT_SECRETS_FILE = 'credentials.json'
    
    creds = None
    
    # Load existing tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Create flow exactly like Flask does
            flow = Flow.from_client_secrets_file(
                CLIENT_SECRETS_FILE,
                scopes=SCOPES,
                redirect_uri='http://localhost:8080/oauth2callback'  # Exact Flask URI
            )
            
            # Generate authorization URL
            authorization_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            print("Please visit this URL to authorize the application:")
            print(authorization_url)
            print("\nWaiting for authorization...")
            
            # Start a local server to handle the callback
            server = HTTPServer(('localhost', 8080), OAuthCallbackHandler)
            server.authorization_response = None
            
            # Open the browser automatically
            webbrowser.open(authorization_url)
            
            # Wait for the callback
            while server.authorization_response is None:
                server.handle_request()
            
            # Use the callback to get tokens
            authorization_response = f"http://localhost:8080{server.authorization_response}"
            flow.fetch_token(authorization_response=authorization_response)
            creds = flow.credentials
            
            server.server_close()
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('gmail', 'v1', credentials=creds)

def main():
    try:
        service = authenticate_gmail_exact_flask_style()
        print("✅ Authentication successful!")
        
        # Test the service
        profile = service.users().getProfile(userId='me').execute()
        print(f"✅ Connected to: {profile['emailAddress']}")
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")

if __name__ == '__main__':
    main()

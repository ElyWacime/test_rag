import os
import pickle
import json
import flask
import google.oauth2.credentials
import google_auth_oauthlib.flow
import googleapiclient.discovery
from googleapiclient.errors import HttpError

# Configuration
CLIENT_SECRETS_FILE = "credentials.json"
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
API_SERVICE_NAME = 'gmail'
API_VERSION = 'v1'

app = flask.Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

@app.route('/')
def index():
    if 'credentials' in flask.session:
        return '''
        <h1>Gmail Export</h1>
        <a href="/export">Export All Emails</a><br>
        <a href="/revoke">Revoke Permissions</a><br>
        <a href="/clear">Clear Session</a>
        '''
    return '<a href="/authorize">Authorize Gmail Access</a>'

@app.route('/authorize')
def authorize():
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES)
    
    # This must match exactly one of the authorized redirect URIs
    flow.redirect_uri = flask.url_for('oauth2callback', _external=True)
    
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent')
    
    flask.session['state'] = state
    return flask.redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    state = flask.session['state']
    
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
    flow.redirect_uri = flask.url_for('oauth2callback', _external=True)
    
    authorization_response = flask.request.url
    flow.fetch_token(authorization_response=authorization_response)
    
    credentials = flow.credentials
    flask.session['credentials'] = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
    
    return flask.redirect('/')

@app.route('/export_all')
def export_all_emails():
    if 'credentials' not in flask.session:
        return flask.redirect('/authorize')
    
    credentials_dict = flask.session['credentials']
    credentials = google.oauth2.credentials.Credentials(
        token=credentials_dict['token'],
        refresh_token=credentials_dict.get('refresh_token'),
        token_uri=credentials_dict['token_uri'],
        client_id=credentials_dict['client_id'],
        client_secret=credentials_dict['client_secret'],
        scopes=credentials_dict['scopes']
    )
    
    try:
        service = googleapiclient.discovery.build(
            API_SERVICE_NAME, API_VERSION, credentials=credentials)
        
        # Create download folder with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        download_folder = f'gmail_export_{timestamp}'
        os.makedirs(download_folder, exist_ok=True)
        
        # Get all message IDs first
        print("Fetching all message IDs...")
        all_message_ids = []
        page_token = None
        
        while True:
            result = service.users().messages().list(
                userId='me',
                pageToken=page_token,
                maxResults=500
            ).execute()
            
            if 'messages' in result:
                message_ids = [msg['id'] for msg in result['messages']]
                all_message_ids.extend(message_ids)
                print(f"Found {len(all_message_ids)} messages so far...")
            
            page_token = result.get('nextPageToken')
            if not page_token:
                break
        
        print(f"Total messages to download: {len(all_message_ids)}")
        
        # Download all messages
        success_count = 0
        error_count = 0
        error_log = []
        
        for i, message_id in enumerate(all_message_ids):
            try:
                # Get full message
                message = service.users().messages().get(
                    userId='me', 
                    id=message_id, 
                    format='raw'
                ).execute()
                
                # Decode and save as .eml
                import base64
                msg_str = base64.urlsafe_b64decode(message['raw'].encode('ASCII'))
                
                filename = f"email_{i+1:06d}_{message_id}.eml"
                filepath = os.path.join(download_folder, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(msg_str)
                
                success_count += 1
                
                # Progress reporting
                if (i + 1) % 50 == 0:
                    progress = (i + 1) / len(all_message_ids) * 100
                    print(f"Progress: {i + 1}/{len(all_message_ids)} ({progress:.1f}%) - {success_count} successful, {error_count} errors")
                
                # Rate limiting
                import time
                time.sleep(0.1)
                
            except Exception as e:
                error_count += 1
                error_log.append(f"Message {message_id}: {str(e)}")
                print(f"Error with message {message_id}: {e}")
                continue
        
        # Save error log
        if error_log:
            with open(os.path.join(download_folder, 'error_log.txt'), 'w') as f:
                f.write('\n'.join(error_log))
        
        # Save summary
        summary = f"""
        Gmail Export Summary
        ===================
        Export time: {datetime.datetime.now()}
        Total messages found: {len(all_message_ids)}
        Successfully downloaded: {success_count}
        Failed downloads: {error_count}
        Download location: {download_folder}
        """
        
        with open(os.path.join(download_folder, 'export_summary.txt'), 'w') as f:
            f.write(summary)
        
        return f'''
        <h1>Export Complete!</h1>
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <p><strong>Total messages found:</strong> {len(all_message_ids)}</p>
            <p><strong>Successfully downloaded:</strong> {success_count}</p>
            <p><strong>Failed downloads:</strong> {error_count}</p>
            <p><strong>Download location:</strong> {download_folder}/</p>
            {'<p style="color: red;"><strong>Some emails failed to download. Check error_log.txt</strong></p>' if error_log else ''}
            <br>
            <a href="/">Back to Home</a>
        </div>
        '''
        
    except Exception as e:
        return f'An error occurred: {str(e)}'

@app.route('/export')
def export_emails():
    if 'credentials' not in flask.session:
        return flask.redirect('/authorize')
    
    credentials_dict = flask.session['credentials']
    credentials = google.oauth2.credentials.Credentials(
        token=credentials_dict['token'],
        refresh_token=credentials_dict.get('refresh_token'),
        token_uri=credentials_dict['token_uri'],
        client_id=credentials_dict['client_id'],
        client_secret=credentials_dict['client_secret'],
        scopes=credentials_dict['scopes']
    )
    
    try:
        # Build Gmail service
        service = googleapiclient.discovery.build(
            API_SERVICE_NAME, API_VERSION, credentials=credentials)
        
        # Get all messages
        messages = []
        page_token = None
        
        while True:
            result = service.users().messages().list(
                userId='me',
                pageToken=page_token,
                maxResults=100
            ).execute()
            
            if 'messages' in result:
                messages.extend(result['messages'])
            
            page_token = result.get('nextPageToken')
            if not page_token:
                break
        
        # Download emails
        download_folder = 'exported_emails'
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
        
        for i, msg in enumerate(messages[:10]):  # Limit to 10 for demo
            try:
                message = service.users().messages().get(
                    userId='me', id=msg['id'], format='full').execute()
                
                # Save message data as JSON
                filename = f"email_{i+1}_{msg['id']}.json"
                filepath = os.path.join(download_folder, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(message, f, indent=2, ensure_ascii=False)
                
            except HttpError as error:
                print(f'Error downloading message: {error}')
                continue
        
        return f'Successfully exported {len(messages[:10])} emails to {download_folder}/'
        
    except HttpError as error:
        return f'An error occurred: {error}'

@app.route('/revoke')
def revoke():
    if 'credentials' not in flask.session:
        return 'No credentials to revoke'
    
    credentials_dict = flask.session['credentials']
    credentials = google.oauth2.credentials.Credentials(
        token=credentials_dict['token']
    )
    
    revoke_response = google.auth.transport.requests.Request().post(
        'https://oauth2.googleapis.com/revoke',
        params={'token': credentials.token},
        headers={'content-type': 'application/x-www-form-urlencoded'}
    )
    
    if revoke_response.status_code == 200:
        del flask.session['credentials']
        return 'Credentials revoked successfully'
    else:
        return 'Error revoking credentials'

@app.route('/clear')
def clear_credentials():
    if 'credentials' in flask.session:
        del flask.session['credentials']
    return 'Credentials cleared'

if __name__ == '__main__':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    app.run('localhost', 8080, debug=True)

from auth_helper import authenticate_gmail_flask_style

def test_auth():
    print("Testing authentication...")
    try:
        service = authenticate_gmail_flask_style()
        print("✅ Authentication successful!")
        
        # Test a simple API call
        profile = service.users().getProfile(userId='me').execute()
        print(f"✅ Connected to: {profile['emailAddress']}")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

if __name__ == '__main__':
    test_auth()

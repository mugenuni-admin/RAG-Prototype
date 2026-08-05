import yaml
import streamlit_authenticator as stauth

def add_user(username, name, email, password):
    with open('auth.yaml') as file:
        config = yaml.safe_load(file)
        
    if 'usernames' not in config['credentials']:
        config['credentials']['usernames'] = {}
        
    if username in config['credentials']['usernames']:
        print(f"User '{username}' already exists.")
        return
        
    # Use standard Hasher to hash the password
    hashed_password = stauth.Hasher.hash(password)
    
    config['credentials']['usernames'][username] = {
        'name': name,
        'email': email,
        'password': hashed_password,
        'failed_login_attempts': 0,
        'logged_in': False
    }
    
    with open('auth.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
        
    print(f"Successfully added user '{username}'.")

if __name__ == "__main__":
    print("--- Add New Investor to Data Room ---")
    username = input("Enter username (e.g., investor1): ")
    name = input("Enter full name: ")
    email = input("Enter email address: ")
    password = input("Enter password: ")
    
    add_user(username, name, email, password)

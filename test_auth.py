import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

with open('auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

print("Hash from YAML:", config['credentials']['usernames']['admin']['password'])

# Let's check it directly
is_valid = stauth.Hasher.check_pw('admin', config['credentials']['usernames']['admin']['password'])
print("Is 'admin' password valid?:", is_valid)

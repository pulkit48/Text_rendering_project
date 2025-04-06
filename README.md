
import os
import requests

# Create directory if it doesn't exist
os.makedirs('/mnt/home-ldap/bansal_ldap/pulkit/pulkit_text/lib/python3.12/site-packages/hpsv2/src/open_clip/', exist_ok=True)

# Download the BPE vocabulary file from OpenCLIP's repository
url = 'https://github.com/mlfoundations/open_clip/raw/main/src/open_clip/bpe_simple_vocab_16e6.txt.gz'
bpe_path = '/mnt/home-ldap/bansal_ldap/pulkit/pulkit_text/lib/python3.12/site-packages/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz'

# Download the file
response = requests.get(url)
with open(bpe_path, 'wb') as f:
    f.write(response.content)

print(f"Downloaded BPE vocabulary file to {bpe_path}")

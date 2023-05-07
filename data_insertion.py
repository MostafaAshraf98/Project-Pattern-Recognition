import os
import gdown

url = 'https://drive.google.com/uc?id=1JLxhdIddq6_vKlHml7jT48VaeXoJjvpR'
output_filename = 'data.zip'

# Get current working directory
cwd = os.getcwd()

# Concatenate current working directory and output filename
output_path = os.path.join(cwd, output_filename)

# Download file from Google Drive to output path
gdown.download(url, output_path, quiet=False)

import zipfile

with zipfile.ZipFile(output_filename, 'r') as zip_ref:
    zip_ref.extractall('./data')


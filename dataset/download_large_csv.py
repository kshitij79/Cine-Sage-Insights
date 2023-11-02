import sys
import os
import json

def download_data(csv_file_name):
    if os.path.exists(csv_file_name):
        return
    
    json_key = csv_file_name.replace('.', '_') + '_url'
    with open('dataset_download_links.json') as f:
        url = json.load(f)[json_key]

    print("Downloading from ", url)
    # Install wget if not present

    os.system(f'wget -O {csv_file_name} --no-check-certificate "{url}"')

if __name__ == '__main__':
    csv_file_name = sys.argv[1]
    download_data(csv_file_name)

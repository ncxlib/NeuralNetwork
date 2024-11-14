import os
import pickle
import urllib
import gzip
from tqdm import tqdm

CACHE_DIR = os.path.expanduser("~/.ncxlib/data")
os.makedirs(CACHE_DIR, exist_ok=True) 

def download(url, name):
    cache_path = os.path.join(CACHE_DIR, f"ncxlib.{name}.data.gz")
    
    if os.path.exists(cache_path):
        print(f"Loading {name} dataset from cache...")
        with gzip.open(cache_path, "rb") as file:
            return file.read()
    
    response = urllib.request.urlopen(url)
    total_size = int(response.headers.get('content-length', 0))
    
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {name} dataset")
    data = b''
    
    while True:
        chunk = response.read(1024) 
        if not chunk:
            break
        data += chunk
        progress_bar.update(len(chunk))
    
    progress_bar.close()
    
    with gzip.open(cache_path, "wb") as file:
        file.write(data)
    
    return data

def load_data(url: str, name: str):
    file_data = download(url, name)
    
    print(f"Decompressing and Loading {name} data..")
    decompressed_data = gzip.decompress(file_data)
    data = pickle.loads(decompressed_data)
    
    return data